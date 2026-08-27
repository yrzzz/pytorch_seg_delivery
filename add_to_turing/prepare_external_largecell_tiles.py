#!/usr/bin/env python
"""Prepare Turing-production large-cell tiles for an external segmentation model.

This reproduces the CPU ``SEGMENT_CELLS`` large-cell input path through the point
where the frozen TensorFlow network would normally be invoked.  In particular it
uses the production outer ROI chunks, inner image tiles, image-scale correction,
bilinear resize, background-aware normalization graph, and edge remainder tiles.

The normalized model inputs are written to disk so a PyTorch environment can
replace only the neural-network forward pass without being imported into Turing.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf
from tensorflow.core.framework import graph_pb2  # pylint: disable=no-name-in-module

from turing.engine.cell_segmentation import cell_segmentation as cs_mod
from turing.engine.cell_segmentation.cell_segmentation_structs import (
    NORMALIZATION_CPU_NAME,
    CellSegConstant,
)
from turing.engine.cell_segmentation.io import validate_tile_overlap
from turing.engine.cell_segmentation.stages.segment_cells import BOUNDARY_SEG_PARAM
from turing.engine.cell_segmentation.stages.shared_stage_fn import (
    get_roi_bbox_list,
    prepare_cell_stain_image,
)
from turing.engine.image.image_utils import (
    get_pixel_offset_from_ome_tiff,
    get_pixel_size_from_ome_tiff,
    read_tiff_metadata,
)
from turing.engine.infra.slices.archives.image_library_support import (
    construct_image_tiles,
)


def load_production_preprocessor():
    """Return resize + normalization implemented with Turing's frozen CPU graph."""
    normalization_graph = graph_pb2.GraphDef()
    graph_path = os.path.join(os.path.dirname(cs_mod.__file__), NORMALIZATION_CPU_NAME)
    with open(graph_path, "rb") as handle:
        normalization_graph.ParseFromString(handle.read())

    @tf.function(
        input_signature=[
            tf.TensorSpec([1, None, None, 2], tf.float32),
            tf.TensorSpec([], tf.float32),
            tf.TensorSpec([], tf.float32),
            tf.TensorSpec([], tf.float32),
            tf.TensorSpec([], tf.int32),
            tf.TensorSpec([], tf.int32),
        ]
    )
    def preprocess(image, percentile, background, ceil, target_height, target_width):
        resized = tf.cond(
            tf.logical_and(
                tf.equal(tf.shape(image)[1], target_height),
                tf.equal(tf.shape(image)[2], target_width),
            ),
            true_fn=lambda: image,
            false_fn=lambda: tf.image.resize(
                image,
                [target_height, target_width],
                method=tf.image.ResizeMethod.BILINEAR,
                preserve_aspect_ratio=False,
            ),
        )
        norm, offset = tf.graph_util.import_graph_def(
            normalization_graph,
            name="",
            input_map={
                "args_tf_0": resized,
                "args_tf_1": percentile,
                "args_tf_2": background,
                "args_tf_3": ceil,
            },
            return_elements=["Identity_2:0", "Identity_3:0"],
        )
        return (resized - offset) / norm

    return preprocess


def inference_shape(height: int, width: int, scale_factor: float) -> tuple[int, int]:
    """Match ``PathSegmenter._cal_resize_shape`` exactly."""
    desired_height = scale_factor * height
    desired_width = scale_factor * width
    target_height = max(16, int(desired_height - (desired_height % 16)))
    target_width = max(16, int(desired_width - (desired_width % 16)))
    return target_height, target_width


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare production-equivalent Turing large-cell model inputs"
    )
    parser.add_argument("--morphology-focus-multifile", required=True)
    parser.add_argument("--boundary-channel-name", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--segmentor-scale", type=float, default=0.25)
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--limit-chunks", type=int, default=None)
    parser.add_argument("--limit-tiles", type=int, default=None)
    args = parser.parse_args()

    source_dir = str(Path(args.morphology_focus_multifile).resolve())
    out_dir = Path(args.out).resolve()
    inputs_dir = out_dir / "normalized_inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    preprocess = load_production_preprocessor()

    tiff_metadata = read_tiff_metadata(source_dir)
    pixel_size = float(get_pixel_size_from_ome_tiff(source_dir))
    # First return value of production ``compute_scaled_area_limit``.
    image_scale = pixel_size / 0.2
    segmentor_scale = float(args.segmentor_scale)
    inference_scale = segmentor_scale * image_scale
    if 0.8 < inference_scale < 1.2:
        inference_scale = 1.0

    outer_rois = get_roi_bbox_list(
        image_height=tiff_metadata.image_height,
        image_width=tiff_metadata.image_width,
        image_tile_height=tiff_metadata.tile_height,
        image_tile_width=tiff_metadata.tile_width,
        scale_factor=segmentor_scale,
        gpu_mode=False,
    )
    selected_outer_rois = outer_rois
    if args.limit_chunks is not None:
        selected_outer_rois = selected_outer_rois[: max(0, args.limit_chunks)]

    pixel_value_offset = get_pixel_offset_from_ome_tiff(
        source_dir, args.boundary_channel_name
    )
    inner_tile_height = int(CellSegConstant.tile_height / segmentor_scale)
    inner_tile_width = int(CellSegConstant.tile_width / segmentor_scale)
    requested_overlap = int(CellSegConstant.tile_overlap / segmentor_scale)
    min_area_px = int(BOUNDARY_SEG_PARAM.min_area / segmentor_scale**2)
    max_area_px = int(BOUNDARY_SEG_PARAM.max_area / segmentor_scale**2)

    rows: list[dict[str, object]] = []
    tile_id = 0
    stop = False
    chunk_ceil_values: dict[str, float] = {}
    for chunk_id, roi_bbox in enumerate(selected_outer_rois):
        composed_image, chunk_orig_x, chunk_orig_y = prepare_cell_stain_image(
            morphology_focus_multifile_dir=source_dir,
            stain_analyte_name=args.boundary_channel_name,
            roi_bbox_list=[roi_bbox],
        )
        if pixel_value_offset:
            composed_image[..., 0] = (
                np.maximum(composed_image[..., 0], pixel_value_offset)
                - pixel_value_offset
            )
        chunk_ceil = float(
            max(np.max(np.percentile(composed_image, 99, axis=(0, 1))), 1000)
        )
        chunk_ceil_values[str(chunk_id)] = chunk_ceil
        inner_overlap = validate_tile_overlap(
            composed_image.shape[0], composed_image.shape[1], requested_overlap
        )
        tile_info_map = construct_image_tiles(
            im_height=composed_image.shape[0],
            im_width=composed_image.shape[1],
            tile_height=inner_tile_height,
            tile_width=inner_tile_width,
            tile_overlap=inner_overlap,
        )
        print(
            f"[chunk] {chunk_id + 1}/{len(selected_outer_rois)} roi={roi_bbox} "
            f"shape={composed_image.shape} ceil={chunk_ceil:.3f} "
            f"inner_tiles={len(tile_info_map)}",
            flush=True,
        )

        for tile_info in tile_info_map.values():
            raw_image = composed_image[
                tile_info.y : tile_info.y + tile_info.dy,
                tile_info.x : tile_info.x + tile_info.dx,
                :,
            ]
            if not np.any(raw_image):
                continue
            target_height, target_width = inference_shape(
                tile_info.dy, tile_info.dx, inference_scale
            )
            normalized = preprocess(
                tf.convert_to_tensor(raw_image[np.newaxis], dtype=tf.float32),
                tf.constant(args.percentile, tf.float32),
                tf.constant(BOUNDARY_SEG_PARAM.background_intensity, tf.float32),
                tf.constant(chunk_ceil, tf.float32),
                tf.constant(target_height, tf.int32),
                tf.constant(target_width, tf.int32),
            ).numpy()[0]
            input_path = inputs_dir / f"tile_{tile_id:05d}.npy"
            np.save(input_path, np.asarray(normalized, dtype=np.float32))
            rows.append(
                {
                    "tile_id": tile_id,
                    "chunk_id": chunk_id,
                    "input_path": str(input_path),
                    "x": chunk_orig_x + tile_info.x,
                    "y": chunk_orig_y + tile_info.y,
                    "raw_height": tile_info.dy,
                    "raw_width": tile_info.dx,
                    "target_height": target_height,
                    "target_width": target_width,
                    "inner_overlap": inner_overlap,
                }
            )
            tile_id += 1
            print(f"[prepare] {tile_id} tiles", flush=True)
            if args.limit_tiles is not None and tile_id >= max(0, args.limit_tiles):
                stop = True
                break
        del composed_image
        if stop:
            break

    manifest_path = out_dir / "tiles.csv"
    fieldnames = [
        "tile_id",
        "chunk_id",
        "input_path",
        "x",
        "y",
        "raw_height",
        "raw_width",
        "target_height",
        "target_width",
        "inner_overlap",
    ]
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "pipeline": "turing_cpu_segment_cells_large_cell_pre_inference",
        "morphology_focus_multifile": source_dir,
        "boundary_channel_name": args.boundary_channel_name,
        "full_height": tiff_metadata.image_height,
        "full_width": tiff_metadata.image_width,
        "source_tile_height": tiff_metadata.tile_height,
        "source_tile_width": tiff_metadata.tile_width,
        "pixel_size_um": pixel_size,
        "pixel_value_offset": pixel_value_offset,
        "image_scale": image_scale,
        "segmentor_scale": segmentor_scale,
        "inference_scale": inference_scale,
        "percentile": args.percentile,
        "background_intensity": BOUNDARY_SEG_PARAM.background_intensity,
        "inner_tile_height": inner_tile_height,
        "inner_tile_width": inner_tile_width,
        "requested_inner_overlap": requested_overlap,
        "min_area_px": min_area_px,
        "max_area_px": max_area_px,
        "expansion_dist": BOUNDARY_SEG_PARAM.expansion_dist,
        "alignment_threshold": BOUNDARY_SEG_PARAM.score_threshold,
        "num_outer_chunks": len(outer_rois),
        "outer_rois": [list(roi) for roi in outer_rois],
        "processed_outer_chunks": len(chunk_ceil_values),
        "chunk_ceil_values": chunk_ceil_values,
        "num_prepared_tiles": len(rows),
    }
    metadata_path = out_dir / "metadata.json"
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"[done] manifest: {manifest_path}")
    print(f"[done] metadata: {metadata_path}")


if __name__ == "__main__":
    main()
