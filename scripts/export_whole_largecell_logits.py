#!/usr/bin/env python
"""Export tiled PyTorch Ridgepath logits from one pyramidal whole-slide OME-TIFF.

This is intentionally separate from ``export_logits.py`` (the validation-manifest
exporter).  It reads two channels from a native TIFF pyramid level, runs overlapping
tiles through a trained segmentation checkpoint, and records the whole-image tile
coordinates needed by Turing to decode and stitch the instances.

The expected model input order is [boundary, DAPI].  The expected model output order
is [semantic, row0:4, col0:4].  Outputs remain raw logits: no sigmoid or softmax is
applied.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import tifffile
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.seg_dataset import normalize_positive  # noqa: E402
from models.encoder_decoder import build_seg_model  # noqa: E402
from train_seg import load_config  # noqa: E402


def tile_starts(size: int, tile_size: int, overlap: int) -> list[int]:
    """Return starts that cover an axis, anchoring the last full tile at the edge."""
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if not 0 <= overlap < tile_size:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_size")
    if size <= tile_size:
        return [0]
    step = tile_size - overlap
    starts = list(range(0, size - tile_size + 1, step))
    final_start = size - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def read_pyramid_channels(
    ome_tiff: str, level: int, boundary_channel: int, dapi_channel: int
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[int, int]]:
    """Load only the requested channel pages from one native TIFF pyramid level."""
    with tifffile.TiffFile(ome_tiff) as tif:
        series = tif.series[0]
        if not 0 <= level < len(series.levels):
            raise ValueError(
                f"pyramid level {level} is unavailable; found {len(series.levels)} levels"
            )
        base_shape = tuple(int(v) for v in series.levels[0].shape[-2:])
        level_shape = tuple(int(v) for v in series.levels[level].shape[-2:])
        pages = series.levels[0].pages
        max_channel = max(boundary_channel, dapi_channel)
        if max_channel >= len(pages):
            raise ValueError(
                f"requested channel {max_channel}, but the OME series has {len(pages)} channels"
            )
        # Resolve the physical file for each channel while the multifile OME handles
        # are open.  Read each file independently below; that avoids tifffile retaining
        # a TiffFrame backed by an already-closed secondary multifile handle.
        boundary_path = str(pages[boundary_channel].parent.filehandle.path)
        dapi_path = str(pages[dapi_channel].parent.filehandle.path)
        print(
            f"[image] loading pyramid level {level} {level_shape}; "
            f"boundary=C{boundary_channel} ({Path(boundary_path).name}), "
            f"DAPI=C{dapi_channel} ({Path(dapi_path).name})"
        )
    with tifffile.TiffFile(boundary_path, _multifile=False) as tif:
        boundary = tif.series[0].levels[level].asarray()
    with tifffile.TiffFile(dapi_path, _multifile=False) as tif:
        dapi = tif.series[0].levels[level].asarray()
    if boundary.shape != level_shape or dapi.shape != level_shape:
        raise ValueError(
            f"unexpected channel shapes: boundary={boundary.shape}, DAPI={dapi.shape}, "
            f"expected={level_shape}"
        )
    return boundary, dapi, base_shape, level_shape


def build_model(config_path: str, checkpoint_path: str, device: torch.device):
    """Reconstruct the training architecture and restore the complete checkpoint."""
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
        description="Export overlapping whole-image PyTorch segmentation logits"
    )
    parser.add_argument("--image", required=True, help="multifile pyramidal OME-TIFF")
    parser.add_argument("--config", required=True, help="training run config.json or YAML")
    parser.add_argument("--checkpoint", required=True, help="trained .pth checkpoint")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--level", type=int, default=2, help="native TIFF pyramid level")
    parser.add_argument("--boundary-channel", type=int, default=1)
    parser.add_argument("--dapi-channel", type=int, default=0)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    parser.add_argument(
        "--amp-dtype", choices=("none", "bf16", "fp16"), default="bf16"
    )
    parser.add_argument("--limit", type=int, default=None, help="smoke-test tile limit")
    parser.add_argument(
        "--start-index", type=int, default=0, help="start at this row-major tile index"
    )
    args = parser.parse_args()

    image_path = str(Path(args.image).resolve())
    config_path = str(Path(args.config).resolve())
    checkpoint_path = str(Path(args.checkpoint).resolve())
    for path in (image_path, config_path, checkpoint_path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    out_dir = Path(args.out).resolve()
    logits_dir = out_dir / "logits_batches"
    logits_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    model, cfg, checkpoint = build_model(config_path, checkpoint_path, device)
    if cfg["in_chans"] != 2 or cfg["num_classes"] != 9:
        raise ValueError(
            f"expected a 2-input/9-output Ridgepath model, got "
            f"in_chans={cfg['in_chans']} num_classes={cfg['num_classes']}"
        )
    print(
        f"[model] {checkpoint_path} epoch={checkpoint.get('epoch')} "
        f"encoder={cfg['encoder']} device={device}"
    )

    boundary, dapi, base_shape, level_shape = read_pyramid_channels(
        image_path, args.level, args.boundary_channel, args.dapi_channel
    )
    level_height, level_width = level_shape
    y_starts = tile_starts(level_height, args.tile_size, args.overlap)
    x_starts = tile_starts(level_width, args.tile_size, args.overlap)
    coordinates = [(x, y) for y in y_starts for x in x_starts]
    coordinates = coordinates[max(0, args.start_index) :]
    if args.limit is not None:
        coordinates = coordinates[: max(0, args.limit)]
    print(
        f"[tiles] {len(y_starts)} rows x {len(x_starts)} columns = "
        f"{len(y_starts) * len(x_starts)} total; exporting {len(coordinates)}"
    )

    if args.amp_dtype == "bf16":
        amp_dtype = torch.bfloat16
    elif args.amp_dtype == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = None

    manifest_rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for batch_start in range(0, len(coordinates), args.batch_size):
            batch_coords = coordinates[batch_start : batch_start + args.batch_size]
            tiles = []
            for x, y in batch_coords:
                boundary_tile = boundary[y : y + args.tile_size, x : x + args.tile_size]
                dapi_tile = dapi[y : y + args.tile_size, x : x + args.tile_size]
                image = np.stack((boundary_tile, dapi_tile), axis=-1)
                image = normalize_positive(image, percentile=args.percentile)
                tiles.append(np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32))
            batch_tensor = torch.from_numpy(np.stack(tiles)).to(device, non_blocking=True)
            amp_context = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if device.type == "cuda" and amp_dtype is not None
                else nullcontext()
            )
            with amp_context:
                output = model(batch_tensor)
            logits = (
                output.permute(0, 2, 3, 1)
                .contiguous()
                .float()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            batch_id = batch_start // args.batch_size
            batch_path = logits_dir / f"logits_{batch_id:05d}.npy"
            np.save(batch_path, logits)
            for local_index, (x, y) in enumerate(batch_coords):
                manifest_rows.append(
                    {
                        "tile_id": batch_start + local_index,
                        "batch_path": str(batch_path),
                        "batch_index": local_index,
                        "x": x,
                        "y": y,
                        "height": args.tile_size,
                        "width": args.tile_size,
                    }
                )
            done = batch_start + len(batch_coords)
            print(f"[export] {done}/{len(coordinates)} tiles", flush=True)

    manifest_path = out_dir / "tiles.csv"
    fieldnames = [
        "tile_id",
        "batch_path",
        "batch_index",
        "x",
        "y",
        "height",
        "width",
    ]
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    metadata = {
        "image_path": image_path,
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "encoder": cfg["encoder"],
        "channel_order": ["boundary", "DAPI"],
        "boundary_channel": args.boundary_channel,
        "dapi_channel": args.dapi_channel,
        "pyramid_level": args.level,
        "base_height": base_shape[0],
        "base_width": base_shape[1],
        "level_height": level_shape[0],
        "level_width": level_shape[1],
        "level_to_base_scale_y": base_shape[0] / level_shape[0],
        "level_to_base_scale_x": base_shape[1] / level_shape[1],
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "percentile": args.percentile,
        "num_tiles": len(manifest_rows),
        "logits_dtype": "float32",
        "logits_channel_order": [
            "semantic",
            "row0",
            "row1",
            "row2",
            "row3",
            "col0",
            "col1",
            "col2",
            "col3",
        ],
    }
    metadata_path = out_dir / "metadata.json"
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"[done] manifest: {manifest_path}")
    print(f"[done] metadata: {metadata_path}")


if __name__ == "__main__":
    main()
