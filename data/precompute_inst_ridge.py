"""Offline inst_ridge precompute (run in the REFERENCE / bazel env).

  RF=<...>/train.runfiles/com_github_10XDev_tenxnet
  PYTHONPATH="$RF" <bazel-py> precompute_inst_ridge.py --limit 16

For each (image.tif, label.pb) pair: decode the .pb instance mask and run
``instance_to_inst_ridge`` (skimage skeletonize + networkx -- the offline step, run ONCE here
in the exact tenxnet env). Caches each inst_ridge as ``<cache>/inst_ridge/<subdir>/<base>.npy``
and writes a manifest CSV. The torch_ssl training Dataset then loads (image, cached inst_ridge)
and runs only the pure-Cython inst_ridge->ridgepath online. tenxnet is NOT modified.
"""
import argparse
import csv
import os
from pathlib import Path

import numpy as np

from tenxnet.vision.dataloaders.data_format.pb_label_util import pb_file_to_mask
from tenxnet.vision.representation.ridgepath_label import instance_to_inst_ridge

ROOT = ("/mnt/deck/1/ruizhi.yuan/tenxnet_deployed_data/large_cell_boundary_full/"
        "large_cell_boundary_v1")
REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE = str(REPO_ROOT / "cache")


def list_pairs(root=ROOT):
    """All (subdir, base, image_path, label_path) with both .tif image and .pb label present."""
    pairs = []
    img_root = os.path.join(root, "images")
    lbl_root = os.path.join(root, "labels")
    for subdir in sorted(os.listdir(img_root)):
        idir, ldir = os.path.join(img_root, subdir), os.path.join(lbl_root, subdir)
        if not os.path.isdir(idir):
            continue
        for f in sorted(os.listdir(idir)):
            if not f.endswith(".tif"):
                continue
            base = f[:-4]
            lp = os.path.join(ldir, base + ".pb")
            if os.path.exists(lp):
                pairs.append((subdir, base, os.path.join(idir, f), lp))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=16, help="number of tiles (tiny subset first)")
    ap.add_argument("--root", default=ROOT, help="dataset root with images/ + labels/ subdirs")
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--nshards", type=int, default=1, help="total shards (for parallel runs)")
    ap.add_argument("--shard", type=int, default=0, help="this shard index (0..nshards-1)")
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                    help="recompute even if a cache .npy already exists")
    ap.add_argument("--anno-names", nargs="*", default=["cell", "large_cell"],
                    help="annotation names rasterized into the instance mask "
                         "(default: cell large_cell -- matches ruizhi/configs/test.yaml; "
                         "excludes 'nucleus' etc.)")
    ap.add_argument("--all-annos", action="store_true",
                    help="rasterize ALL annotations (anno_names=None; includes nucleus)")
    args = ap.parse_args()
    manifest = args.manifest or os.path.join(args.cache, "manifest.csv")
    anno = None if args.all_annos else args.anno_names

    pairs = list_pairs(args.root)[: args.limit]
    shard_pairs = pairs[args.shard:: args.nshards] if args.nshards > 1 else pairs
    print(f"found {len(pairs)} pairs; shard {args.shard}/{args.nshards} -> {len(shard_pairs)} tiles "
          f"| anno_names={anno}")
    rows = []
    for i, (subdir, base, img_path, lbl_path) in enumerate(shard_pairs):
        out_dir = os.path.join(args.cache, "inst_ridge", subdir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, base + ".npy")
        if args.skip_existing and os.path.exists(out_path):
            rows.append(dict(image_path=img_path, inst_ridge_path=out_path,
                             subdir=subdir, base=base))
            continue
        inst = pb_file_to_mask(lbl_path, semantic=False, anno_names=anno).astype(np.uint16)  # (H,W)
        sem = (inst > 0).astype(np.uint16)
        instance_label = np.stack([inst, sem], axis=-1)                     # (H,W,2)
        inst_ridge = instance_to_inst_ridge(instance_label).astype(np.uint16)  # (H,W,3)
        np.save(out_path, inst_ridge)
        rows.append(dict(image_path=img_path, inst_ridge_path=out_path,
                         subdir=subdir, base=base))
        if (i + 1) % 10 == 0 or i + 1 == len(shard_pairs):
            print(f"  [shard {args.shard}] {i+1}/{len(shard_pairs)} done", flush=True)

    os.makedirs(os.path.dirname(manifest), exist_ok=True)
    with open(manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["image_path", "inst_ridge_path", "subdir", "base"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} cache files + manifest {manifest}")


if __name__ == "__main__":
    main()
