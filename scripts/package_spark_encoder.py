"""Package a SparK timm-style ``tenxnet_recipe`` encoder export into a checkpoint that
``restore_seg_encoder`` (init: seg_encoder) can load into the seg model with 0 missing keys.

SparK writes ``<exp_dir>/tenxnet_recipe_1kpretrained_timm_style.pth`` (and epoch snapshots
``tenxnet_recipe_ep{5,10,25,50,100}_timm_style.pth``) as a BARE encoder state_dict whose keys match
``build_encoder("tenxnet_recipe", 2, encoder_norm="bn")`` exactly (stem.*/groups.*, 138 tensors). This
just verifies that parity and re-wraps as ``{"model": {"encoder."+k: v}, "epoch": N}`` -- the format
``restore_seg_encoder`` expects (mirrors cache/fcmae_5stage_atto_seg_encoder.pth).

    python scripts/package_spark_encoder.py \
        --src /mnt/home/ruizhi.yuan/SparK/runs/tenxnet_recipe_spark_bn/tenxnet_recipe_ep100_timm_style.pth \
        --out cache/tenxnet_recipe_spark_seg_encoder.pth --epoch 100
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.encoders import build_encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="SparK timm-style export (bare encoder state_dict)")
    ap.add_argument("--out", required=True, help="output .pth for init: seg_encoder")
    ap.add_argument("--epoch", type=int, default=0)
    args = ap.parse_args()

    sd = torch.load(args.src, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:      # tolerate an already-wrapped file
        sd = {k[len("encoder."):] if k.startswith("encoder.") else k: v for k, v in sd["model"].items()}

    ref = build_encoder("tenxnet_recipe", 2, encoder_norm="bn").state_dict()
    ks, kr = set(sd), set(ref)
    missing = sorted(kr - ks)      # in seg encoder, absent from export -> would stay random
    unexpected = sorted(ks - kr)   # in export, absent from seg encoder
    mism = [k for k in ks & kr if tuple(sd[k].shape) != tuple(ref[k].shape)]
    assert not missing, f"{len(missing)} missing encoder keys, e.g. {missing[:8]}"
    assert not unexpected, f"{len(unexpected)} unexpected keys, e.g. {unexpected[:8]}"
    assert not mism, f"{len(mism)} shape mismatches, e.g. {mism[:8]}"

    out = {"model": {f"encoder.{k}": v for k, v in sd.items()}, "epoch": args.epoch}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(out, args.out)
    print(f"[package] {len(sd)} encoder tensors -> {args.out}")
    print(f"[package] 0 missing / 0 unexpected / 0 mismatched vs tenxnet_recipe (encoder_norm=bn)")


if __name__ == "__main__":
    main()
