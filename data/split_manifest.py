"""Deterministic, leakage-safe train/val split of the full seg manifest.

Groups tiles by **source-slide prefix** (everything before ``_x_`` in the base name) so crops
from the same slide never span train/val (prevents spatial leakage). Deterministic: groups are
sorted then shuffled with a fixed seed; whole groups are assigned to val until ~val_frac of
tiles is reached. Writes ``manifest_train.csv`` + ``manifest_val.csv`` next to the input.

  python split_manifest.py --full cache/manifest_full.csv --val_frac 0.15 --seed 0
"""
import argparse
import csv
import os
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def slide_group(base: str) -> str:
    """`hu_breast_1583096_1643209_x_11306_y_9230_s_0.25` -> `hu_breast_1583096_1643209`."""
    i = base.find("_x_")
    return base[:i] if i > 0 else base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default=str(REPO_ROOT / "cache/manifest_full.csv"))
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.full, newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames
        rows = list(reader)
    if not rows:
        raise SystemExit(f"empty manifest: {args.full}")

    groups = {}
    for r in rows:
        groups.setdefault(slide_group(r["base"]), []).append(r)
    gkeys = sorted(groups)
    random.Random(args.seed).shuffle(gkeys)

    target_val = args.val_frac * len(rows)
    val_groups, n_val = set(), 0
    for g in gkeys:
        if n_val < target_val:
            val_groups.add(g)
            n_val += len(groups[g])

    train = [r for r in rows if slide_group(r["base"]) not in val_groups]
    val = [r for r in rows if slide_group(r["base"]) in val_groups]
    assert len(train) + len(val) == len(rows)
    assert not ({slide_group(r["base"]) for r in train} & {slide_group(r["base"]) for r in val}), \
        "slide leakage between train and val!"

    out_dir = os.path.dirname(args.full)
    for name, subset in (("manifest_train.csv", train), ("manifest_val.csv", val)):
        with open(os.path.join(out_dir, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(subset)

    print(f"total tiles: {len(rows)} | slide-groups: {len(groups)}")
    print(f"train: {len(train)} tiles ({len(groups) - len(val_groups)} slides) -> manifest_train.csv")
    print(f"val:   {len(val)} tiles ({len(val_groups)} slides) -> manifest_val.csv "
          f"({len(val)/len(rows):.1%})")
    print("val slide-groups:", sorted(val_groups))


if __name__ == "__main__":
    main()
