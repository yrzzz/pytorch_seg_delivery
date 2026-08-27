#!/usr/bin/env python
"""Run a PyTorch Ridgepath network on Turing-prepared large-cell tiles.

All image reading, resizing, normalization, tiling, decoding, and stitching are
owned by Turing companion scripts.  This program intentionally performs only the
external neural-network forward pass and records raw nine-channel logits.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.encoder_decoder import build_seg_model  # noqa: E402
from train_seg import load_config  # noqa: E402


def build_model(config_path: str, checkpoint_path: str, device: torch.device):
    cfg = load_config(config_path, {})
    model_kwargs = dict(cfg.get("model_kwargs") or {})
    if cfg["encoder"] in ("convnext_dino", "convnext_dino_hires", "hires_only"):
        model_kwargs["pretrained"] = False
    model = build_seg_model(
        encoder_name=cfg["encoder"],
        in_chans=cfg["in_chans"],
        num_classes=cfg["num_classes"],
        encoder_norm=cfg.get("encoder_norm", "bn"),
        gn_max_groups=cfg.get("gn_max_groups", 32),
        **model_kwargs,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, cfg, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infer raw PyTorch logits on production-prepared Turing tiles"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--amp-dtype", choices=("none", "bf16", "fp16"), default="none"
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reuse existing logits whose shape and dtype are valid (default: true)",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    metadata_path = Path(args.metadata).resolve()
    config_path = str(Path(args.config).resolve())
    checkpoint_path = str(Path(args.checkpoint).resolve())
    out_dir = Path(args.out).resolve()
    logits_dir = out_dir / "logits"
    logits_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]
    if not rows:
        raise ValueError("prepared-tile manifest contains no rows")

    with metadata_path.open() as handle:
        preparation_metadata = json.load(handle)
    if preparation_metadata.get("pipeline") != (
        "turing_cpu_segment_cells_large_cell_pre_inference"
    ):
        raise ValueError("metadata was not produced by the Turing production preparer")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    model, cfg, checkpoint = build_model(config_path, checkpoint_path, device)
    if cfg["in_chans"] != 2 or cfg["num_classes"] != 9:
        raise ValueError(
            "production Ridgepath interchange requires a 2-input/9-output model; "
            f"found {cfg['in_chans']}/{cfg['num_classes']}"
        )

    if args.amp_dtype == "bf16":
        amp_dtype = torch.bfloat16
    elif args.amp_dtype == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = None

    output_rows: list[dict[str, object]] = []
    print(
        f"[model] {checkpoint_path} epoch={checkpoint.get('epoch')} "
        f"encoder={cfg['encoder']} device={device} amp={args.amp_dtype}",
        flush=True,
    )
    with torch.inference_mode():
        for index, row in enumerate(rows):
            expected_shape = (
                int(row["target_height"]),
                int(row["target_width"]),
                2,
            )
            logits_path = logits_dir / f"tile_{int(row['tile_id']):05d}.npy"
            if args.resume and logits_path.exists():
                existing = np.load(logits_path, mmap_mode="r")
                expected_logits_shape = expected_shape[:2] + (9,)
                if existing.shape == expected_logits_shape and existing.dtype == np.float32:
                    output_row = dict(row)
                    output_row["logits_path"] = str(logits_path)
                    output_rows.append(output_row)
                    print(f"[reuse] {index + 1}/{len(rows)} tiles", flush=True)
                    continue
                print(
                    f"[resume] replacing invalid {logits_path}: "
                    f"shape={existing.shape} dtype={existing.dtype}",
                    flush=True,
                )

            image = np.load(row["input_path"], mmap_mode="r")
            if image.shape != expected_shape:
                raise ValueError(
                    f"tile {row['tile_id']} input {image.shape} != {expected_shape}"
                )
            image_tensor = torch.from_numpy(
                np.asarray(image).transpose(2, 0, 1).copy()
            ).unsqueeze(0)
            image_tensor = image_tensor.to(device, dtype=torch.float32, non_blocking=True)
            amp_context = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if device.type == "cuda" and amp_dtype is not None
                else nullcontext()
            )
            with amp_context:
                output = model(image_tensor)
            logits = (
                output[0]
                .permute(1, 2, 0)
                .contiguous()
                .float()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            if logits.shape != expected_shape[:2] + (9,):
                raise ValueError(
                    f"tile {row['tile_id']} logits {logits.shape} do not match "
                    f"{expected_shape[:2] + (9,)}"
                )
            np.save(logits_path, logits)
            output_row = dict(row)
            output_row["logits_path"] = str(logits_path)
            output_rows.append(output_row)
            print(f"[infer] {index + 1}/{len(rows)} tiles", flush=True)

    output_manifest = out_dir / "tiles.csv"
    fieldnames = list(output_rows[0].keys())
    with output_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    inference_metadata = {
        "pipeline": "external_pytorch_network_only",
        "source_manifest": str(manifest_path),
        "source_metadata": str(metadata_path),
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "encoder": cfg["encoder"],
        "amp_dtype": args.amp_dtype,
        "num_inferred_tiles": len(output_rows),
    }
    output_metadata = out_dir / "metadata.json"
    with output_metadata.open("w") as handle:
        json.dump(inference_metadata, handle, indent=2)
    print(f"[done] manifest: {output_manifest}")
    print(f"[done] metadata: {output_metadata}")


if __name__ == "__main__":
    main()
