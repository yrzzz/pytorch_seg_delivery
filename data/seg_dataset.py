"""PyTorch Dataset for ridgepath cell segmentation (torch_ssl env).

Loads a 2-channel TIFF image + its cached inst_ridge (.npy, precomputed offline in the
reference env by precompute_inst_ridge.py), applies the tenxnet train augmentation pipeline
(ruizhi/configs/test.yaml) in the SAME order, then runs the pure-Cython inst_ridge -> ridgepath
target-gen ONLINE (post-aug). Returns ``(image[2,H,W] float32, target[10,H,W] float32)``.

Augmentation order (faithful to tenxnet):
  image: RandomColorQuantization -> Normalization(positive,p99) -> RandomBrightnessContrast
         -> RandomMirroring -> Random90Rotation
  label (inst_ridge): RandomMirroring -> Random90Rotation  (synchronized with the image)
Photometric ops touch the image only; the geometric ops (mirror=flip-both-axes, 90-deg rot) are
drawn once and applied identically to image + inst_ridge, then targets are generated post-aug.

The online target-gen reuses the validated standalone Cython via the runtime glue
(``inst_ridge_to_ridgepath`` + ``load_ops("standalone")``). The Cython (OpenMP) ops are loaded
LAZILY inside the worker (not the parent) to avoid OpenMP-before-fork issues; set
OMP_NUM_THREADS=1 per worker via ``seg_worker_init_fn`` to avoid oversubscription.
"""
import csv
import os

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

# Verbatim Tenxnet wrappers plus a loader for this repository's standalone Cython extensions.
from runtime.ridgepath_glue import inst_ridge_to_ridgepath, load_ops

# ridgepath target-gen params (from doc/examples/ridgepath_example.yaml / ruizhi/configs/test.yaml)
DEFAULT_PARAMS = dict(dist_cutoff=15.0, weight_sigma=1.0, smooth_range=3, weighted_loss=True)

# tenxnet train-augmentation defaults (ruizhi/configs/test.yaml train_transform)
DEFAULT_AUG = dict(
    color_quant_bins=[64, 128, 256, 512], color_quant_prob=0.5,
    brightness_delta=0.2, contrast_range=(0.8, 1.2),
    mirror_prob=0.5, rot_choices=[0, 90, 180, 270],
)


def normalize_positive(img, percentile=99.0):
    """tenxnet Normalization 'positive': per-channel (x - min) / max(p99_nonzero - min, 1e-3)."""
    img = img.astype(np.float64)
    out = np.empty_like(img)
    for c in range(img.shape[-1]):
        ch = img[..., c]
        lo = float(ch.min())
        nz = ch[ch > 0]
        hi = float(np.percentile(nz, percentile)) if nz.size else 0.0
        norm = hi - lo
        out[..., c] = (ch - lo) / (norm if norm > 1e-3 else 1.0)
    return out


# ---- photometric transforms (image only); use numpy GLOBAL rng -> per-epoch variation via
# ---- seg_worker_init_fn (and the num_workers=0 path reseeds per epoch in the train loop).
def color_quantization(img, num_bins):
    """== transforms.color_quantization: per-channel floor-quantize to ~num_bins levels."""
    max_val = img.max(axis=(0, 1))
    bin_size = np.maximum(max_val // num_bins, 1).astype(img.dtype)
    return (img // bin_size) * bin_size


def aug_color_quantization(img, bins, prob):
    """RandomColorQuantization, applied on the raw uint16 image BEFORE normalization."""
    if np.random.rand() < prob:
        img = color_quantization(img, int(np.random.choice(bins)))
    return img


def aug_brightness_contrast(img, brightness_delta, contrast_range):
    """RandomBrightnessContrast (AFTER normalization): +delta, then per-channel (x-mean)*f+mean.

    Matches tf.image.stateless_random_brightness/contrast (one scalar delta, one scalar factor,
    per-channel mean for contrast; no clipping).
    """
    img = img + np.random.uniform(-brightness_delta, brightness_delta)
    f = np.random.uniform(contrast_range[0], contrast_range[1])
    mean = img.mean(axis=(0, 1), keepdims=True)
    return (img - mean) * f + mean


# ---- geometric transforms (image + label, synchronized: drawn once, applied to both) ----
def sample_geom(mirror_prob, rot_choices):
    """RandomMirroring (flip BOTH axes with prob p) + Random90Rotation (uniform choice)."""
    do_mirror = bool(np.random.rand() < mirror_prob)
    rot_k = int(np.random.choice(rot_choices)) // 90
    return do_mirror, rot_k


def apply_geom(arr, do_mirror, rot_k):
    """Apply the SAME geometric transform to an HWC array (axes 0,1). Contiguous copy."""
    if do_mirror:
        arr = arr[::-1, ::-1]  # flip_left_right + flip_up_down (== 180 deg); matches RandomMirroring
    if rot_k:
        arr = np.rot90(arr, k=rot_k, axes=(0, 1))
    return np.ascontiguousarray(arr)


def seg_worker_init_fn(worker_id):
    """Seed numpy per worker (so geom aug varies) + avoid OpenMP oversubscription."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch as _torch
    np.random.seed(_torch.initial_seed() % 2**32)


class RidgepathSegDataset(Dataset):
    def __init__(self, manifest, augment=True, params=None, percentile=99.0, seed=0, aug=None):
        with open(manifest, newline="") as fh:
            self.rows = list(csv.DictReader(fh))
        self.augment = augment
        self.params = dict(DEFAULT_PARAMS, **(params or {}))
        self.aug = dict(DEFAULT_AUG, **(aug or {}))
        self.percentile = percentile
        self.seed = seed
        self._ops = None  # (lm, rc, cp); lazy-loaded per worker

    def __len__(self):
        return len(self.rows)

    def _get_ops(self):
        if self._ops is None:  # first use happens inside the worker process
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            self._ops = load_ops("standalone", verbose=False)
        return self._ops

    def _target(self, inst_ridge):
        lm, rc, _ = self._get_ops()
        return inst_ridge_to_ridgepath(
            inst_ridge, self.params["dist_cutoff"], self.params["weight_sigma"],
            self.params["smooth_range"], lm, rc, weighted_loss=self.params["weighted_loss"],
        )

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = tifffile.imread(row["image_path"])          # (H,W,2) uint16
        inst_ridge = np.load(row["inst_ridge_path"])         # (H,W,3) uint16
        if image.ndim == 2:
            image = image[..., None]

        a = self.aug
        if self.augment:
            do_mirror, rot_k = sample_geom(a["mirror_prob"], a["rot_choices"])  # drawn once, shared
            # photometric (image only), in tenxnet order: color-quant (pre-norm) -> norm -> bright/contrast
            image = aug_color_quantization(image, a["color_quant_bins"], a["color_quant_prob"])
            image = normalize_positive(image, self.percentile)
            image = aug_brightness_contrast(image, a["brightness_delta"], a["contrast_range"])
            # geometric (synchronized) on image + label, then targets are generated post-aug
            image = apply_geom(image, do_mirror, rot_k)
            inst_ridge = apply_geom(inst_ridge, do_mirror, rot_k)
        else:
            image = normalize_positive(image, self.percentile)
            inst_ridge = np.ascontiguousarray(inst_ridge)

        target = self._target(inst_ridge.astype(np.uint16))  # (H,W,10) float64

        image_t = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
        target_t = torch.from_numpy(np.ascontiguousarray(target.transpose(2, 0, 1))).float()
        return image_t, target_t
