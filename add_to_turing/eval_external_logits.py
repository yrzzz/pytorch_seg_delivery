# Copyright (c) 2025 10x Genomics, Inc. All rights reserved.

"""Offline instance-segmentation eval for EXTERNAL (e.g. PyTorch) model logits.

Reuses turing's canonical decode + metrics so results match production. Given a manifest of per-tile
raw 9-channel logits (NHWC float32 .npy, channel order [semantic, row0:4, col0:4]) exported from an
external model, for each tile this:

  1. runs the frozen post-model graph (9ch -> direct_map / stationary_px),
  2. runs the Rust cellseg_o3 decode (-> instance label_mask),
  3. loads the GT instance mask from the exported `gt/<key>.npy` (inst_ridge ch0 = [cell, large_cell]
     per-cell ids; produced by pytorch_seg/export_logits.py, so no `tenxnet` import is needed here),
  4. scores with calculate_instance_metric,

then aggregates precision / recall / F1 (+ split/merge/miss/extra) like
headless_benchmark_segmentation.py and writes detailed + summary JSON.

This is a standalone research script (no bazel target). Run in the pipeline env:

  bazel build //:devpipes_env
  bazel-bin/devpipes_env.sh python \
    lib/python/turing/analysis/cell_segmentation/eval_external_logits.py \
    --export-manifest <export_manifest.csv> \
    --out-detailed-json detailed.json --out-summary-json summary.json
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.core.framework import graph_pb2  # pylint: disable=no-name-in-module

from turing.analysis.cell_segmentation.segmentation_metrics import calculate_instance_metric
from turing.engine.cell_segmentation import cell_segmentation as cs_mod
from turing.engine.cell_segmentation.cell_segmentation_structs import POST_MODEL_NAME
from turing.engine.cell_segmentation.cellseg_o3 import path_segmenter_post_process


def _relabel_contiguous(labels: np.ndarray) -> np.ndarray:
    """Contiguous 0..K instance labels (background 0 preserved). np-only equivalent of turing's
    clean_up_instance_labels (which lives in benchmark_segmentation, not bundled in devpipes_env)."""
    _, inv = np.unique(labels, return_inverse=True)  # sorted uniques -> 0 (bg, smallest) stays 0
    return inv.reshape(labels.shape).astype(np.uint32)


def load_post_model():
    """Load the frozen post-model graph (9ch logits -> direct_map, stationary_px) as a tf.function.

    Mirrors PathSegmenter.__init__ (cell_segmentation.py:73-75) + the import_graph_def in _infer
    (cell_segmentation.py:179-184): input node "model_output", outputs "Identity:0"/"Identity_1:0".
    """
    post = graph_pb2.GraphDef()
    post_path = os.path.join(os.path.dirname(cs_mod.__file__), POST_MODEL_NAME)
    with open(post_path, "rb") as fh:
        post.ParseFromString(fh.read())

    @tf.function(input_signature=[tf.TensorSpec([1, None, None, 9], tf.float32)])
    def run_post(model_output):
        direct_map, stationary_px = tf.graph_util.import_graph_def(
            post, name="", input_map={"model_output": model_output},
            return_elements=["Identity:0", "Identity_1:0"],
        )
        return direct_map, stationary_px

    return run_post


def decode_logits(run_post, logits_hw9: np.ndarray, p: dict) -> np.ndarray:
    """9-channel logits (H,W,9) -> instance label_mask (H,W) uint32, contiguous.

    Mirrors the canonical call in cell_segmentation.py:257-271 (slice [:, :, :, 0] -> [1,H,W] int8;
    keep label_mask as uint32 -- the production int32 cast is only for the optional TF resize, which we
    skip at native resolution).
    """
    model_output = tf.constant(logits_hw9[None].astype(np.float32))  # [1, H, W, 9]
    direct_map, stationary_px = run_post(model_output)
    direct = direct_map.numpy()[:, :, :, 0].astype(np.int8)          # [1, H, W]
    stationary = stationary_px.numpy()[:, :, :, 0].astype(np.int8)   # [1, H, W]
    label_mask, _score_mapping = path_segmenter_post_process(
        direct,
        stationary,
        merge_threshold=p["merge_threshold"],
        merge_center_radius=p["merge_radius"],
        min_support=p["min_support"],
        correct_dict=p["correct_dict"],
        filter_by_alignment=p["filter_by_alignment"],
        inner_threshold=p["inner_threshold"],
        gradient_threshold=p["gradient_threshold"],
        distance_threshold=p["distance_threshold"],
        num_threads=p["num_threads"],
    )
    return _relabel_contiguous(np.asarray(label_mask).astype(np.uint32))


def load_gt(gt_npy_path: str) -> np.ndarray:
    """GT instance mask (H,W) uint32 from the exported inst_ridge-ch0 .npy (contiguous-relabeled)."""
    gt = np.load(gt_npy_path)
    if gt.ndim == 3:  # (H,W,C) or (C,H,W) -> first channel is the instance channel
        gt = gt[0] if gt.shape[0] <= 3 else gt[..., 0]
    return _relabel_contiguous(gt.astype(np.uint32))


def _save_overlay(
    out_dir: str,
    key: str,
    label_pb_path: str,
    label_mask: np.ndarray,
    gt: np.ndarray,
):
    """Save the boundary image with GT and predicted instance overlays side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tifffile

    img_path = label_pb_path.replace("/labels/", "/images/").rsplit(".", 1)[0] + ".tif"
    bd = tifffile.imread(img_path).astype(np.float32)
    if bd.ndim == 3:
        bd = bd[..., 0] if bd.shape[-1] <= 5 else bd[0]
    lo, hi = np.percentile(bd, 1), np.percentile(bd, 99)
    bd = np.clip((bd - lo) / (hi - lo + 1e-6), 0, 1)

    def make_overlay(mask: np.ndarray, seed: int):
        rng = np.random.default_rng(seed)
        n = int(mask.max())
        lut = np.zeros((n + 1, 3), dtype=np.float32)  # id 0 = black background
        if n:
            lut[1:] = rng.random((n, 3))
        inst_rgb = lut[mask]
        overlay = np.repeat(bd[..., None], 3, axis=2)
        fg = mask > 0
        overlay[fg] = 0.45 * overlay[fg] + 0.55 * inst_rgb[fg]
        return overlay, n

    gt_overlay, n_gt = make_overlay(gt, seed=0)
    pred_overlay, n_pred = make_overlay(label_mask, seed=1)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(bd, cmap="gray"); ax[0].set_title("Boundary image", fontsize=10)
    ax[1].imshow(gt_overlay); ax[1].set_title(f"Ground truth (n={n_gt})", fontsize=10)
    ax[2].imshow(pred_overlay); ax[2].set_title(f"Prediction (n={n_pred})", fontsize=10)
    for a in ax:
        a.axis("off")
    os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{key}.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser("offline instance eval of external 9-ch logits via turing decode+metrics")
    ap.add_argument("--export-manifest", required=True,
                    help="CSV from pytorch_seg/export_logits.py (key,npy_path,label_pb_path,H,W)")
    ap.add_argument("--out-detailed-json", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--tp-threshold", type=float, default=0.5)
    ap.add_argument("--assign-threshold", type=float, default=0.05)
    # decode params -- PathSegmenter defaults (cell_segmentation.py __init__ + run)
    ap.add_argument("--merge-threshold", type=int, default=500)
    ap.add_argument("--merge-radius", type=int, default=3)
    ap.add_argument("--min-support", type=int, default=100)
    ap.add_argument("--correct-dict", action="store_true")
    ap.add_argument("--filter-by-alignment", action="store_true")
    ap.add_argument("--inner-threshold", type=float, default=10.0)
    ap.add_argument("--gradient-threshold", type=float, default=4.0)
    ap.add_argument("--distance-threshold", type=float, default=15.0)
    ap.add_argument("--num-threads", type=int, default=1,
                    help="must be 1 -- the Python cellseg_o3 binding rejects >1 thread")
    ap.add_argument("--save-overlay-dir", default=None, help="if set, dump colored instance overlays")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N tiles")
    args = ap.parse_args()

    p = dict(merge_threshold=args.merge_threshold, merge_radius=args.merge_radius,
             min_support=args.min_support, correct_dict=args.correct_dict,
             filter_by_alignment=args.filter_by_alignment, inner_threshold=args.inner_threshold,
             gradient_threshold=args.gradient_threshold, distance_threshold=args.distance_threshold,
             num_threads=args.num_threads)

    run_post = load_post_model()
    with open(args.export_manifest, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit is not None:
        rows = rows[: args.limit]
    print(f"[eval] {len(rows)} tiles | decode={p} "
          f"| tp={args.tp_threshold} assign={args.assign_threshold}")

    detailed = {}
    tp_sum = fp_sum = fn_sum = 0
    for i, r in enumerate(rows):
        logits = np.load(r["npy_path"])  # (H, W, 9) float32
        pred = decode_logits(run_post, logits, p)
        gt = load_gt(r["gt_npy_path"])
        if gt.shape != pred.shape:
            raise ValueError(f"{r['key']}: GT shape {gt.shape} != pred shape {pred.shape}")
        m = calculate_instance_metric(gt, pred, tp_threshold=args.tp_threshold,
                                      assign_threshold=args.assign_threshold)
        d = dataclasses.asdict(m)
        d["n_gt"] = int(gt.max())
        d["n_pred"] = int(pred.max())
        detailed[r["key"]] = d
        tp_sum += m.true_positive
        fp_sum += m.false_positive
        fn_sum += m.false_negative
        if args.save_overlay_dir:
            _save_overlay(args.save_overlay_dir, r["key"], r["label_pb_path"], pred, gt)
        print(f"  [{i + 1}/{len(rows)}] {r['key']}: TP={m.true_positive} FP={m.false_positive} "
              f"FN={m.false_negative} (gt={d['n_gt']} pred={d['n_pred']})")

    precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) else 1.0
    recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    summary = {
        "n_tiles": len(rows), "tp": tp_sum, "fp": fp_sum, "fn": fn_sum,
        "precision": precision, "recall": recall, "f1": f1,
        "tp_threshold": args.tp_threshold, "assign_threshold": args.assign_threshold,
        "decode_params": p,
    }
    # calculate_instance_metric returns numpy int64 fields (via dataclasses.asdict); default=int
    # coerces those (and any np scalars in summary) so json.dump doesn't choke.
    with open(args.out_detailed_json, "w") as fh:
        json.dump(detailed, fh, indent=2, default=int)
    with open(args.out_summary_json, "w") as fh:
        json.dump(summary, fh, indent=2, default=int)
    print(f"[eval] overall: precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} "
          f"(TP={tp_sum} FP={fp_sum} FN={fn_sum})")
    print(f"[eval] wrote {args.out_summary_json} + {args.out_detailed_json}")


if __name__ == "__main__":
    main()
