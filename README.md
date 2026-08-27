# PyTorch Ridgepath segmentation

Training and inference code for two-channel Ridgepath cell segmentation. The model consumes
`[boundary, DAPI]` images and emits nine raw logits: one semantic channel, four row-direction
channels, and four column-direction channels.

The production workflow uses this repository for model inference and the separate Turing repository
for image preparation, Ridgepath instance decoding, tile merging, and Zarr creation.

## Repository layout

```text
configs/
  convnext_dino/       active ConvNeXt-DINOv3 and high-resolution experiments
  tenxnet_recipe/      Tenxnet-recipe, DINO/iBOT, and SparK experiments
  legacy/              ResNet18 and FCMAE experiments
cython_ops/             standalone Ridgepath target-generation extensions
data/                   datasets and manifest/target preprocessing
losses/                 Ridgepath training loss
models/                 encoders, FPN, and segmentation heads
runtime/                training-time Ridgepath glue and extension loader
scripts/                inference, export, packaging, and Turing launchers
reports/                retained analysis notebooks
train_seg.py            training entry point
restore.py              checkpoint restoration helpers
```

## Environment

Use a Python environment with a CUDA-compatible PyTorch build when training on GPU, then install the
remaining packages from `requirements.txt`.

Build the standalone Ridgepath extensions once:

```bash
cd cython_ops
python setup.py build_ext --inplace
cd ..
```

Training loads these extensions from this repository through `runtime/ridgepath_glue.py`; it does
not import code from another `pytorch_seg` checkout.

## Data and artifact paths

Config paths under `cache/` and `runs/` are resolved relative to the repository by default. Set
`PYTORCH_SEG_ARTIFACT_ROOT` to use manifests and checkpoints stored elsewhere:

```bash
export PYTORCH_SEG_ARTIFACT_ROOT=/path/to/pytorch_seg_artifacts
```

That directory should contain the referenced `cache/` manifests/checkpoints and receives configured
`runs/` outputs. Absolute paths in a config remain unchanged.

Important: all CSV manifests under `pytorch_seg_delivery/cache/` contain environment-specific
locations in their `image_path` and `inst_ridge_path` columns. Before training or exporting on a
new machine, update both columns in every manifest to point to the corresponding image and
`inst_ridge` data directories on that machine. Copying the manifests does not rewrite these paths
automatically.

## Pipeline

1. Convert instance-label protobufs to cached `inst_ridge.npy` arrays with
   `data/precompute_inst_ridge.py` in the Tenxnet reference environment.
2. During training, `data/seg_dataset.py` augments the image and `inst_ridge` together, then creates
   the ten-channel Ridgepath target after geometric augmentation.
3. `train_seg.py` trains the encoder, FPN, and segmentation head and saves the resolved config with
   each checkpoint.
4. Turing prepares normalized production tiles. `scripts/infer_prepared_largecell_tiles.py` loads
   the saved config/checkpoint and exports `[H,W,9]` raw logits.
5. Turing decodes the logits into instances, filters and merges overlapping cells, and builds the
   final Zarr output.

## Current-node patch-evaluation example

This example performs a 16-sample `inst_ridge` smoke test, builds the exact Tenxnet split from the
existing complete `pytorch_seg_cache_cell` cache, trains the ConvNeXt-DINOv3 model, exports
validation-patch logits, and evaluates them with Turing. On a new server, remove `--limit 16` and
point `--cache` and `--inst-ridge-root` to the same full cache to generate everything from scratch.

### 1. Create `inst_ridge`

Run the offline protobuf conversion with Tenxnet's Bazel Python and runfiles:

```bash
SEG_REPO=/mnt/home/ruizhi.yuan/pytorch_seg_delivery
RAW_ROOT=/mnt/deck/2/ruizhi.yuan/tenxnet_deployed_data/cell_boundary/v1
OUTPUT_ROOT=/mnt/deck/2/ruizhi.yuan/precompute_test

BAZEL_PYTHON=/mnt/bazelbuild/user/ruizhi.yuan/66d30966b7045ad5bca10aeb4ea3520e/execroot/com_github_10XDev_tenxnet/bazel-out/k8-fastbuild/bin/external/anaconda/bin/python3
TENXNET_RUNFILES=/mnt/home/ruizhi.yuan/tenxnet/bazel-bin/bin/train.runfiles/com_github_10XDev_tenxnet

mkdir -p "${OUTPUT_ROOT}"
cd "${SEG_REPO}"

PYTHONPATH="${TENXNET_RUNFILES}" "${BAZEL_PYTHON}" \
  data/precompute_inst_ridge.py \
  --root "${RAW_ROOT}" \
  --cache "${OUTPUT_ROOT}" \
  --manifest "${OUTPUT_ROOT}/manifest_full.csv" \
  --limit 16 \
  --anno-names cell large_cell
```

### 2. Build the exact Tenxnet split

```bash
cd /mnt/home/ruizhi.yuan/pytorch_seg_delivery

/mnt/home/ruizhi.yuan/miniforge/miniforge3/envs/torch_ssl/bin/python \
  data/build_tenxnet_exact_split.py \
  --images-root /mnt/deck/2/ruizhi.yuan/tenxnet_deployed_data/cell_boundary/v1/images \
  --inst-ridge-root /mnt/deck/2/ruizhi.yuan/pytorch_seg_cache_cell/inst_ridge \
  --train-split 0.9 \
  --val-split 0.1 \
  --seed 42 \
  --out-train /mnt/deck/2/ruizhi.yuan/precompute_test/manifest_cell_tenxnet_train.csv \
  --out-val /mnt/deck/2/ruizhi.yuan/precompute_test/manifest_cell_tenxnet_val.csv \
  --out-dropped /mnt/deck/2/ruizhi.yuan/precompute_test/manifest_cell_tenxnet_dropped.csv
```

Install the generated train/validation manifests at the paths expected by the training config:

```bash
mkdir -p /mnt/home/ruizhi.yuan/pytorch_seg_delivery/cache
cp /mnt/deck/2/ruizhi.yuan/precompute_test/manifest_cell_tenxnet_train.csv \
  /mnt/home/ruizhi.yuan/pytorch_seg_delivery/cache/
cp /mnt/deck/2/ruizhi.yuan/precompute_test/manifest_cell_tenxnet_val.csv \
  /mnt/home/ruizhi.yuan/pytorch_seg_delivery/cache/
```

### 3. Train

```bash
cd /mnt/home/ruizhi.yuan/pytorch_seg_delivery

CUDA_VISIBLE_DEVICES=0 torchrun train_seg.py \
  --config configs/convnext_dino/convnext_dino_hires_unfrozen_normal_cell.yaml \
  --ddp
```

### 4. Export validation-patch logits

```bash
cd /mnt/home/ruizhi.yuan/pytorch_seg_delivery

python scripts/export_logits.py \
  --config /mnt/home/ruizhi.yuan/pytorch_seg_delivery/runs/convnext_dino_hires_unfrozen_normal_cell_tenxnet_split/config.json \
  --checkpoint /mnt/home/ruizhi.yuan/pytorch_seg_delivery/runs/convnext_dino_hires_unfrozen_normal_cell_tenxnet_split/best.pth \
  --manifest /mnt/home/ruizhi.yuan/pytorch_seg_delivery/cache/manifest_cell_tenxnet_val.csv \
  --out /mnt/home/ruizhi.yuan/pytorch_seg_delivery/runs/convnext_dino_hires_unfrozen_normal_cell_tenxnet_split/turing_val \
  --device cuda
```

### 5. Decode and evaluate in Turing

The external-logit bridge script must be present in the Turing checkout.

```bash
cd /mnt/home/ruizhi.yuan/turing

bazel-bin/devpipes_env.sh python \
  lib/python/turing/analysis/cell_segmentation/eval_external_logits.py \
  --export-manifest /mnt/home/ruizhi.yuan/pytorch_seg_delivery/runs/convnext_dino_hires_unfrozen_normal_cell_tenxnet_split/turing_val/export_manifest.csv \
  --out-detailed-json /mnt/home/ruizhi.yuan/pytorch_seg_delivery/runs/convnext_dino_hires_unfrozen_normal_cell_tenxnet_split/turing_val/turing_detailed.json \
  --out-summary-json /mnt/home/ruizhi.yuan/pytorch_seg_delivery/runs/convnext_dino_hires_unfrozen_normal_cell_tenxnet_split/turing_val/turing_summary.json \
  --save-overlay-dir /mnt/home/ruizhi.yuan/pytorch_seg_delivery/runs/convnext_dino_hires_unfrozen_normal_cell_tenxnet_split/turing_val/overlays \
  --merge-threshold 2000 \
  --merge-radius 3 \
  --min-support 200 \
  --correct-dict \
  --filter-by-alignment \
  --inner-threshold 6 \
  --gradient-threshold 4 \
  --distance-threshold 15
```

## Training

ConvNeXt-DINOv3 high-resolution example:

```bash
python train_seg.py \
  --config configs/convnext_dino/convnext_dino_hires_unfrozen_normal_cell.yaml
```

Tenxnet-recipe SparK example:

```bash
python train_seg.py \
  --config configs/tenxnet_recipe/tenxnet_recipe_spark_unfrozen.yaml
```

`scripts/package_spark_encoder.py` validates a bare SparK timm-style encoder export and wraps it in
the checkpoint format expected by segmentation configs that use `init: seg_encoder`:

```bash
python scripts/package_spark_encoder.py \
  --src /mnt/home/ruizhi.yuan/SparK/runs/tenxnet_recipe_spark_bn_1x/tenxnet_recipe_ep1600_timm_style.pth \
  --out /mnt/home/ruizhi.yuan/pytorch_seg/cache/tenxnet_recipe_spark_1x_seg_encoder.pth \
  --epoch 1600
```

Useful options include `--smoke`, `--epochs`, `--batch-size`, `--num-workers`, `--out-dir`,
`--resume`, `--no-amp`, and `--max-iters`. Use `torchrun ... --ddp` for multi-GPU training.

## Patch-level evaluation

Turing evaluation requires the bridge scripts in `add_to_turing/`; install them using
`add_to_turing/README.md`.

Export raw logits and instance-label ground truth for a validation manifest:

```bash
python scripts/export_logits.py \
  --config runs/<run>/config.json \
  --checkpoint runs/<run>/best.pth \
  --manifest /path/to/manifest_cell_tenxnet_val.csv \
  --out runs/<run>/turing_val \
  --device cuda
```

The output contains `logits/`, `gt/`, and `export_manifest.csv` for Turing instance evaluation.
Use `--n 16` for a small export or omit it to export the full validation split.

## Production Turing workflow

The sample launcher connects Turing preparation, this repository's PyTorch inference, Turing decode,
and final Zarr construction:

```bash
PYTORCH_SEG_ARTIFACT_ROOT=/path/to/artifacts \
bash scripts/run_hu_muscle_whole_largecell.sh
```

The launcher also accepts environment overrides:

- `PYTORCH_SEG_RUN_DIR`: directory containing `config.json` and `best.pth`
- `PYTORCH_SEG_TURING_REPO`: Turing checkout
- `PYTORCH_SEG_PYTHON`: Python executable for PyTorch inference
- `PYTORCH_SEG_SAMPLE_DIR`: input sample directory
- `PYTORCH_SEG_OUTPUT_DIR`: workflow output directory

For direct control, invoke `scripts/infer_prepared_largecell_tiles.py` with Turing's prepared
manifest and metadata. `scripts/export_whole_largecell_logits.py` is the alternative standalone
OME-TIFF tiling exporter.

## External dependencies

- Turing supplies production image preparation, decoding, stitching, and Zarr construction.
- Tenxnet is required only for the offline `.pb` to `inst_ridge.npy` preprocessing step and optional
  Bazel-reference comparisons.
- Pretrained ConvNeXt-DINOv3 initialization may require cached or downloadable timm weights. Export
  and inference disable pretrained downloads because all model weights come from the checkpoint.
