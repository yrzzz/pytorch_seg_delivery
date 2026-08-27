"""Stratified subsample of a seg manifest: keep a fraction of the tiles from EVERY slide.

Groups rows by ``subdir`` (== source slide) and keeps ``max(1, round(frac * n))`` tiles per group, so
every slide stays represented. A flat random sample over the whole manifest would leave the smallest
slides with zero tiles (this train manifest has slides with as few as 14), which turns a
"fewer labels per slide" ablation into a "fewer slides" one by accident.

Output preserves the input's column order and row order; only rows are dropped. Deterministic for a
given ``--seed``.

Run:  python data/subsample_manifest.py \
          --src cache/manifest_cell_tenxnet_train.csv \
          --out cache/manifest_cell_tenxnet_train_10pct.csv --frac 0.1 --seed 0
"""
import argparse
import collections
import csv
import os
import random


def subsample(src, out, frac, seed=0, group_key="subdir"):
    with open(src, newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if group_key not in (fieldnames or []):
        raise ValueError(f"{src} has no {group_key!r} column (columns: {fieldnames})")
    if not 0 < frac <= 1:
        raise ValueError(f"--frac must be in (0, 1], got {frac}")

    # group row INDICES so the output can keep the manifest's original ordering
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows):
        groups[r[group_key]].append(i)

    rng = random.Random(seed)
    keep = set()
    for g in sorted(groups):                       # sorted -> seed alone fixes the result
        idxs = groups[g]
        k = max(1, round(frac * len(idxs)))         # every slide keeps >= 1 tile
        keep.update(rng.sample(idxs, k))

    kept = [rows[i] for i in range(len(rows)) if i in keep]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    per_slide = collections.Counter(r[group_key] for r in kept)
    sizes = sorted(per_slide.values())
    print(f"src  {src}: {len(rows)} tiles / {len(groups)} slides")
    print(f"out  {out}: {len(kept)} tiles / {len(per_slide)} slides "
          f"({len(kept) / len(rows):.2%} of source, frac={frac}, seed={seed})")
    print(f"     tiles/slide kept: min={sizes[0]} median={sizes[len(sizes) // 2]} max={sizes[-1]}")
    return kept


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source manifest CSV")
    ap.add_argument("--out", required=True, help="destination manifest CSV")
    ap.add_argument("--frac", type=float, default=0.1, help="fraction of tiles to keep per slide")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--group-key", default="subdir", help="column identifying the source slide")
    args = ap.parse_args()
    subsample(args.src, args.out, args.frac, args.seed, args.group_key)
