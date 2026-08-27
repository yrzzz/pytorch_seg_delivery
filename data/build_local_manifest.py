#!/usr/bin/env python
"""Build train/val seg manifests from LOCAL data (home dir), avoiding the broken /mnt/deck mount.

Pairs each local image tile (~/data/.../images/<subdir>/<base>.tif) with its already-precomputed
inst_ridge label (~/pytorch_seg/cache/inst_ridge/<subdir>/<base>.npy). Emits the standard seg-manifest
columns [image_path, inst_ridge_path, subdir, base], then does a leakage-safe train/val split
(whole source-slide groups, i.e. everything before ``_x_``, never span train/val).

  python data/build_local_manifest.py \
    --images-root ~/data/large_cell_boundary/large_cell_boundary/images \
    --inst-ridge-root cache/inst_ridge --val-frac 0.1 --seed 42 \
    --out-train cache/manifest_train_local.csv --out-val cache/manifest_val_local.csv
"""
import argparse
import csv
import os
import random


def slide_group(base: str) -> str:
    """`hu_breast_1583096_1643209_x_11306_y_9230_s_0.25` -> `hu_breast_1583096_1643209` (no crop leakage)."""
    i = base.find("_x_")
    return base[:i] if i > 0 else base


def main():
    ap = argparse.ArgumentParser("build local seg train/val manifests")
    ap.add_argument("--images-root", default=os.path.expanduser(
        "~/data/large_cell_boundary/large_cell_boundary/images"))
    ap.add_argument("--inst-ridge-root", default="cache/inst_ridge")
    ap.add_argument("--val-frac", type=float, default=0.1)   # == his train_split 0.9
    ap.add_argument("--seed", type=int, default=42)          # == his data_split_rng_seed 42
    ap.add_argument("--out-train", default="cache/manifest_train_local.csv")
    ap.add_argument("--out-val", default="cache/manifest_val_local.csv")
    args = ap.parse_args()

    images_root = os.path.abspath(os.path.expanduser(args.images_root))
    ir_root = os.path.abspath(os.path.expanduser(args.inst_ridge_root))
    rows, missing = [], []
    for subdir in sorted(os.listdir(images_root)):
        sub_img = os.path.join(images_root, subdir)
        if not os.path.isdir(sub_img):
            continue
        for fn in sorted(os.listdir(sub_img)):
            if not fn.endswith(".tif"):
                continue
            base = fn[:-4]  # strip ".tif" (keeps the "s_0.25" suffix intact)
            ir_path = os.path.join(ir_root, subdir, base + ".npy")
            if not os.path.exists(ir_path):
                missing.append(f"{subdir}/{base}")
                continue
            rows.append(dict(image_path=os.path.join(sub_img, fn), inst_ridge_path=ir_path,
                             subdir=subdir, base=base))

    if not rows:
        raise SystemExit(f"no paired tiles found under {images_root} (+ {ir_root})")
    if missing:
        print(f"[warn] {len(missing)} images without a matching inst_ridge .npy (skipped), "
              f"e.g. {missing[:3]}")

    # leakage-safe split: shuffle slide-groups, fill val to ~val_frac of tiles
    groups = {}
    for r in rows:
        groups.setdefault(slide_group(r["base"]), []).append(r)
    gkeys = sorted(groups)
    random.Random(args.seed).shuffle(gkeys)
    target_val, val_groups, n_val = args.val_frac * len(rows), set(), 0
    for g in gkeys:
        if n_val < target_val:
            val_groups.add(g)
            n_val += len(groups[g])
    train = [r for r in rows if slide_group(r["base"]) not in val_groups]
    val = [r for r in rows if slide_group(r["base"]) in val_groups]
    assert len(train) + len(val) == len(rows)
    assert not ({slide_group(r["base"]) for r in train} & val_groups), "slide leakage!"

    fields = ["image_path", "inst_ridge_path", "subdir", "base"]
    for path, subset in ((args.out_train, train), (args.out_val, val)):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(subset)

    per_sub = {}
    for r in rows:
        per_sub[r["subdir"]] = per_sub.get(r["subdir"], 0) + 1
    print(f"total paired tiles: {len(rows)}  per-subdir: {per_sub}")
    print(f"train: {len(train)} tiles ({len(groups) - len(val_groups)} slides) -> {args.out_train}")
    print(f"val:   {len(val)} tiles ({len(val_groups)} slides, {len(val)/len(rows):.1%}) -> {args.out_val}")


if __name__ == "__main__":
    main()
