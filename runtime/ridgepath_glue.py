"""Parity glue: load the Ridgepath Cython ops from EITHER build, and run the exact
tenxnet target-generation / decode wrappers on them.

The two glue functions below are verbatim copies of:
  - tenxnet/vision/representation/ridgepath_label.py :: inst_ridge_to_ridgepath
  - tenxnet/vision/post_analysis/centerpath_post.py  :: cp_construct_instance
with the only change being that the Cython callables are passed in (so the SAME glue
runs against the standalone build or the bazel build). This makes the comparison a clean
"identical glue + identical inputs, two Cython builds" test.
"""
import glob
import importlib.util
import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
STANDALONE_DIR = REPO_ROOT / "cython_ops"
BAZEL_VISION = Path(os.environ.get(
    "TENXNET_BAZEL_VISION", "/mnt/home/ruizhi.yuan/tenxnet/bazel-bin/tenxnet/vision"
))


def _load_by_path(modname, path):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resolve(build):
    if build == "standalone":
        g = lambda n: sorted(glob.glob(os.path.join(STANDALONE_DIR, n + "*.so")))[0]
        return {
            "label_morph": g("label_morph"),
            "ridgepath_construct": g("ridgepath_construct"),
            "centerpath_post_cy": g("centerpath_post_cy"),
        }
    if build == "tenxnet":
        return {
            "label_morph": f"{BAZEL_VISION}/post_analysis/label_morph.so",
            "ridgepath_construct": f"{BAZEL_VISION}/representation/ridgepath_construct.so",
            "centerpath_post_cy": f"{BAZEL_VISION}/post_analysis/centerpath_post_cy.so",
        }
    raise ValueError(f"unknown build {build!r}")


def load_ops(build, verbose=True):
    """Return (label_morph, ridgepath_construct, centerpath_post_cy) for the given build."""
    paths = _resolve(build)
    if verbose:
        print(f"[{build}] loading:")
        for k, v in paths.items():
            print(f"    {k}: {v}")
    lm = _load_by_path("label_morph", paths["label_morph"])
    rc = _load_by_path("ridgepath_construct", paths["ridgepath_construct"])
    cp = _load_by_path("centerpath_post_cy", paths["centerpath_post_cy"])
    return lm, rc, cp


# ----- verbatim glue (Cython callables injected as lm / rc / cp) -----


def inst_ridge_to_ridgepath(  # pylint: disable=too-many-arguments
    inst_ridge_lb, dist_cutoff, weight_sigma, smooth_range, lm, rc, weighted_loss=True
):
    """Copy of tenxnet ridgepath_label.inst_ridge_to_ridgepath (Cython injected)."""
    row, col, _ = inst_ridge_lb.shape
    ridgepath_lb = np.zeros((row, col, 10))
    ridgepath_lb[:, :, 0] = inst_ridge_lb[:, :, 1]
    instance_lb = inst_ridge_lb[:, :, 0]
    ridge_lb = inst_ridge_lb[:, :, 2]
    edt_sq = lm.label_edtsq_2d(instance_lb).reshape(instance_lb.shape).astype(np.uint16)
    row_prob, col_prob = rc.label_to_ridge_direct_prob(ridge_lb, instance_lb, edt_sq)
    if weighted_loss:
        weight = 1 / (1 + np.exp((np.sqrt(edt_sq) - dist_cutoff) / weight_sigma))
        mask = row_prob[:, :, 3] == 0
        row_prob[:, :, 0] = np.multiply(row_prob[:, :, 0], weight)
        row_prob[:, :, 2] = np.multiply(row_prob[:, :, 2], weight)
        row_prob[:, :, 1][mask] = (1 - (row_prob[:, :, 0] + row_prob[:, :, 2]))[mask]
        col_prob[:, :, 0] = np.multiply(col_prob[:, :, 0], weight)
        col_prob[:, :, 2] = np.multiply(col_prob[:, :, 2], weight)
        col_prob[:, :, 1][mask] = (1 - (col_prob[:, :, 0] + col_prob[:, :, 2]))[mask]
        ridgepath_lb[:, :, 9] = weight / 2 + 0.5
    else:
        ridgepath_lb[:, :, 9] = 1
    if smooth_range is not None:
        row_prob, col_prob = rc.smooth_direct_prob(
            row_prob, col_prob, instance_lb, ridge_lb, smooth_range
        )
    ridgepath_lb[:, :, 1:5] = row_prob
    ridgepath_lb[:, :, 5:9] = col_prob
    return ridgepath_lb


def cp_construct_instance(  # pylint: disable=too-many-arguments
    row_prob, col_prob, confident_lvl, bg_confidence_lvl, cp, min_support=50, merge_center_radius=3
):
    """Copy of tenxnet centerpath_post.cp_construct_instance (Cython injected; numpy-only path)."""
    height, width, _ = row_prob.shape
    direct_row = np.ones((height, width))
    temp_direct = np.argmax(row_prob[:, :, :3], axis=2)
    hi = np.max(row_prob[:, :, :3], axis=2) > confident_lvl
    direct_row[hi] = temp_direct[hi]

    direct_col = np.ones((height, width))
    temp_direct = np.argmax(col_prob[:, :, :3], axis=2)
    hi = np.max(col_prob[:, :, :3], axis=2) > confident_lvl
    direct_col[hi] = temp_direct[hi]

    background = np.logical_or(
        row_prob[:, :, 3] > bg_confidence_lvl, col_prob[:, :, 3] > bg_confidence_lvl
    )
    direct_map = direct_row * 3 + direct_col
    direct_map[background] = -999
    direct_map = direct_map.astype(np.int32).flatten(order="C")
    stationary_px = (direct_map == 4).astype(np.int32) * 2
    label_mask, support_counter, area = cp.direction_to_label(
        direct_map,
        stationary_px=stationary_px,
        height=height,
        width=width,
        background_val=-999,
        min_support=min_support,
        merge_center_radius=merge_center_radius,
    )
    return label_mask.reshape(height, width), np.asarray(support_counter), np.asarray(area)
