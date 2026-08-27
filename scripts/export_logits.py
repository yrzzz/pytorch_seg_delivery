#!/usr/bin/env python
"""Export raw 9-channel segmentation logits for offline instance evaluation in turing.

Decoupled from training. Given a seg config + checkpoint, runs the model over a manifest and saves,
per tile, the RAW 9-channel output as a `[H, W, 9]` float32 NHWC `.npy` (no sigmoid/softmax). These
feed turing's canonical decode + instance metrics (run in turing's devpipes_env; see
turing .../analysis/cell_segmentation/eval_external_logits.py).

Channel order (matches turing + losses/ridgepath_loss.py): [semantic, row0:4, col0:4].

Outputs to <out>/:
  - logits/<key>.npy           (H, W, 9) float32, key = "<subdir>__<base>" (avoids cross-folder collisions)
  - gt/<key>.npy               (H, W) uint32 GT instance mask = inst_ridge ch0 ([cell, large_cell] ids;
                               ch0>0 == semantic target, verified IoU 1.0). Lets turing score GT without
                               importing tenxnet (its devpipes_env has none).
  - export_manifest.csv        columns: key, npy_path, gt_npy_path, label_pb_path, H, W  (absolute paths)

Usage:
  python scripts/export_logits.py --config <cfg> --checkpoint runs/<run>/best.pth \
      --manifest <validation.csv> --out <dir> --n 5 --device cpu
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from models.encoder_decoder import build_seg_model  # noqa: E402
from data.seg_dataset import RidgepathSegDataset  # noqa: E402
from train_seg import load_config  # noqa: E402 (DEFAULTS + YAML merge; no side effects on import)


def _label_pb_path(image_path: str) -> str:
    """Derive the .pb instance-label path from the image .tif path (images/ -> labels/, .tif -> .pb).

    Mirrors how data/precompute_inst_ridge.py pairs each image tile with its label.
    """
    return image_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".pb"


def main():
    ap = argparse.ArgumentParser("export raw 9-ch seg logits (NHWC float32) for turing eval")
    ap.add_argument("--config", required=True, help="seg YAML (model arch + data + out_dir)")
    ap.add_argument("--checkpoint", default=None,
                    help="path to .pth (default: <out_dir>/best.pth, else <out_dir>/checkpoint.pth)")
    ap.add_argument("--manifest", default=None, help="data manifest CSV (default: cfg val_manifest)")
    ap.add_argument("--out", default=None, help="output dir (default: <out_dir>/export)")
    ap.add_argument("--n", type=int, default=None, help="limit to N tiles (default: all)")
    ap.add_argument("--indices", type=int, nargs="+", default=None,
                    help="explicit tile indices (overrides --n)")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    args = ap.parse_args()

    cfg = load_config(args.config, {})
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt = args.checkpoint
    if ckpt is None:
        best = os.path.join(cfg["out_dir"], "best.pth")
        ckpt = best if os.path.exists(best) else os.path.join(cfg["out_dir"], "checkpoint.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    # Reconstruct the exact architecture used for training. Checkpoint-based export does not need
    # timm/HF pretrained weights because the checkpoint immediately supplies the complete state.
    model_kwargs = dict(cfg.get("model_kwargs") or {})
    if cfg["encoder"] in ("convnext_dino", "convnext_dino_hires", "hires_only"):
        model_kwargs["pretrained"] = False
    model = build_seg_model(encoder_name=cfg["encoder"], in_chans=cfg["in_chans"],
                            num_classes=cfg["num_classes"],
                            encoder_norm=cfg.get("encoder_norm", "bn"),
                            gn_max_groups=cfg.get("gn_max_groups", 32),
                            **model_kwargs).to(device)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])  # save_ckpt stores the raw (unwrapped) module state_dict
    model.eval()
    print(f"[export] {ckpt} (epoch {ck.get('epoch')}) | encoder={cfg['encoder']} init={cfg.get('init')} "
          f"| device={device}")

    manifest = args.manifest or cfg["val_manifest"]
    ds = RidgepathSegDataset(manifest, augment=False, params=cfg["target_params"])
    out = os.path.abspath(args.out or os.path.join(cfg["out_dir"], "export"))
    logits_dir = os.path.join(out, "logits")
    gt_dir = os.path.join(out, "gt")
    os.makedirs(logits_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    if args.indices is not None:
        idx = [i for i in args.indices if 0 <= i < len(ds)]
    else:
        idx = list(range(len(ds) if args.n is None else min(args.n, len(ds))))

    manifest_rows = []
    with torch.no_grad():
        for i in idx:
            row = ds.rows[i]
            key = f"{row['subdir']}__{row['base']}"
            image_t, _ = ds[i]                                   # image_t: [2, H, W] float (normalized)
            logits = model(image_t.unsqueeze(0).to(device))[0]   # [9, H, W] raw logits
            nhwc = logits.permute(1, 2, 0).contiguous().cpu().numpy().astype(np.float32)  # [H, W, 9]
            npy_path = os.path.join(logits_dir, f"{key}.npy")
            np.save(npy_path, nhwc)
            # GT instance mask = inst_ridge channel 0 (per-cell ids for [cell, large_cell]; verified
            # ch0>0 == semantic target, IoU 1.0). Saving it here avoids a tenxnet dependency on the
            # turing side (turing's devpipes_env has no `tenxnet`); the driver loads GT from this .npy.
            gt = np.load(row["inst_ridge_path"])[..., 0].astype(np.uint32)  # [H, W]
            gt_path = os.path.join(gt_dir, f"{key}.npy")
            np.save(gt_path, gt)
            H, W = nhwc.shape[0], nhwc.shape[1]
            manifest_rows.append(dict(key=key, npy_path=npy_path, gt_npy_path=gt_path,
                                      label_pb_path=_label_pb_path(row["image_path"]), H=H, W=W))
            if len(manifest_rows) % 20 == 0:
                print(f"  exported {len(manifest_rows)}/{len(idx)}")

    man_path = os.path.join(out, "export_manifest.csv")
    with open(man_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["key", "npy_path", "gt_npy_path", "label_pb_path", "H", "W"])
        w.writeheader()
        w.writerows(manifest_rows)
    print(f"[export] wrote {len(manifest_rows)} logit tiles to {logits_dir}")
    print(f"[export] manifest: {man_path}")


if __name__ == "__main__":
    main()
