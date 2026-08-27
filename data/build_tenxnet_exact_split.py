#!/usr/bin/env python
"""Build PyTorch manifests with Tenxnet's exact per-image-type split.

This mirrors ``tenxnet.vision.dataloaders.dataset_utils``:

* enumerate sorted files independently within each image-type subdirectory;
* reset NumPy's legacy global RNG to the same seed for every image type;
* shuffle a Python list of indices with ``np.random.shuffle``;
* take ``floor(train_split * N)`` train and ``floor(val_split * N)`` val;
* leave any rounding remainder unused.

The output manifests can be consumed directly by ``train_seg.py``.
"""
import argparse
import csv
import os

import numpy as np


FIELDS = ["image_path", "inst_ridge_path", "subdir", "base"]


def _write_manifest(path, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images-root",
        default="/mnt/deck/2/ruizhi.yuan/tenxnet_deployed_data/cell_boundary/v1/images",
    )
    parser.add_argument(
        "--inst-ridge-root",
        default="/mnt/deck/2/ruizhi.yuan/pytorch_seg_cache_cell/inst_ridge",
    )
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-train", default="cache/manifest_cell_tenxnet_train.csv"
    )
    parser.add_argument(
        "--out-val", default="cache/manifest_cell_tenxnet_val.csv"
    )
    parser.add_argument(
        "--out-dropped", default="cache/manifest_cell_tenxnet_dropped.csv"
    )
    args = parser.parse_args()

    if not np.isclose(args.train_split + args.val_split, 1.0):
        raise ValueError("train_split and val_split must sum to 1")

    train_rows, val_rows, dropped_rows = [], [], []
    per_type = []
    image_types = [
        name
        for name in sorted(os.listdir(args.images_root))
        if os.path.isdir(os.path.join(args.images_root, name))
    ]
    if not image_types:
        raise SystemExit(f"no image-type directories under {args.images_root}")

    for image_type in image_types:
        image_dir = os.path.join(args.images_root, image_type)
        image_names = sorted(
            name for name in os.listdir(image_dir) if name.lower().endswith(".tif")
        )
        rows = []
        for image_name in image_names:
            base = os.path.splitext(image_name)[0]
            inst_ridge_path = os.path.join(
                args.inst_ridge_root, image_type, base + ".npy"
            )
            if not os.path.isfile(inst_ridge_path):
                raise FileNotFoundError(
                    f"missing inst_ridge for {image_type}/{image_name}: "
                    f"{inst_ridge_path}"
                )
            rows.append(
                {
                    "image_path": os.path.join(image_dir, image_name),
                    "inst_ridge_path": inst_ridge_path,
                    "subdir": image_type,
                    "base": base,
                }
            )

        # Deliberately reset the legacy NumPy RNG for every image type, exactly
        # as Tenxnet's indices_split() does.
        indices = list(range(len(rows)))
        np.random.seed(args.seed)
        np.random.shuffle(indices)
        train_end = int(np.floor(args.train_split * len(rows)))
        val_end = train_end + int(np.floor(args.val_split * len(rows)))

        train_rows.extend(rows[i] for i in indices[:train_end])
        val_rows.extend(rows[i] for i in indices[train_end:val_end])
        dropped_rows.extend(rows[i] for i in indices[val_end:])
        per_type.append(
            (image_type, len(rows), train_end, val_end - train_end, len(rows) - val_end)
        )

    for path in (args.out_train, args.out_val, args.out_dropped):
        out_dir = os.path.dirname(os.path.abspath(path))
        os.makedirs(out_dir, exist_ok=True)
    _write_manifest(args.out_train, train_rows)
    _write_manifest(args.out_val, val_rows)
    _write_manifest(args.out_dropped, dropped_rows)

    total = len(train_rows) + len(val_rows) + len(dropped_rows)
    print(
        f"image types: {len(per_type)} | total: {total} | "
        f"train: {len(train_rows)} | val: {len(val_rows)} | "
        f"dropped by Tenxnet flooring: {len(dropped_rows)}"
    )
    print(f"train -> {args.out_train}")
    print(f"val   -> {args.out_val}")
    print(f"unused -> {args.out_dropped}")


if __name__ == "__main__":
    main()
