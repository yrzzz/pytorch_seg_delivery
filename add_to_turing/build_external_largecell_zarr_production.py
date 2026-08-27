#!/usr/bin/env python
"""Build a base-resolution production-style cell dataset from large-cell RLEs.

This invokes the same polygon, pyramid, mask, summary, and dataset functions as
``GENERATE_CELLS_DATASET``.  The dataset is deliberately large-cell-only for this
experiment, so the production consolidation with normal cells, interior cells,
and nuclei is outside its scope.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from turing.engine.cell_segmentation.cellseg_o3 import (
    load_rles_from_pb,
    save_rles_to_pb,
)
from turing.engine.cell_segmentation.stages import generate_cells_dataset as stage
from turing.engine.image.image_utils import (
    construct_homography_matrix_from_ome_tiff,
    read_tiff_metadata,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a production-style large-cell-only cells.zarr.zip"
    )
    parser.add_argument("--cell-rles", required=True)
    parser.add_argument("--morphology", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--boundary-analyte", required=True)
    parser.add_argument("--max-vertices", type=int, default=25)
    parser.add_argument("--simplify-epsilon", type=float, default=0.1)
    parser.add_argument("--large-cell-scale", type=float, default=0.25)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--objects-per-chunk", type=int, default=1000)
    args = parser.parse_args()

    cell_rles_path = str(Path(args.cell_rles).resolve())
    morphology_path = str(Path(args.morphology).resolve())
    out_path = Path(args.out).resolve()
    work_dir = Path(args.work_dir).resolve()
    chunks_dir = work_dir / "polygon_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cell_rles = load_rles_from_pb(cell_rles_path)
    num_cells = len(cell_rles)
    if num_cells == 0:
        raise ValueError("cannot build a viewer dataset with zero cells")
    labels = np.asarray([rle.label for rle in cell_rles], dtype=np.uint32)
    if not np.array_equal(labels, np.arange(1, num_cells + 1, dtype=np.uint32)):
        raise ValueError("cell RLE labels must be consecutive and start at 1")

    tiff_metadata = read_tiff_metadata(morphology_path)
    full_height = tiff_metadata.image_height
    full_width = tiff_metadata.image_width
    pixel_to_physical = np.linalg.inv(
        construct_homography_matrix_from_ome_tiff(morphology_path)
    ).astype(np.float32)
    print(
        f"[dataset] {num_cells} large cells at base shape "
        f"{(full_height, full_width)}",
        flush=True,
    )

    nucleus_rles_path = work_dir / "empty_nucleus_rles.pb"
    save_rles_to_pb([], str(nucleus_rles_path))
    nucleus_ranges, cell_ranges = stage.split_start_end_idx(
        num_nucleus=0,
        num_cell=num_cells,
        min_num_obj_per_chunk=args.objects_per_chunk,
    )

    output_lists: dict[str, list[str]] = {
        "nucleus_vertices": [],
        "nucleus_num_vertices": [],
        "nucleus_int_id": [],
        "nucleus_pyramid": [],
        "cell_vertices": [],
        "cell_num_vertices": [],
        "cell_int_id": [],
        "cell_pyramid": [],
        "bboxes": [],
        "z_levels": [],
    }
    for chunk_index, (nucleus_range, cell_range) in enumerate(
        zip(nucleus_ranges, cell_ranges)
    ):
        chunk_dir = chunks_dir / f"chunk_{chunk_index:04d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        paths = {name: str(chunk_dir / f"{name}.npz") for name in output_lists}
        stage.main(
            nucleus_rles_path=str(nucleus_rles_path),
            cell_rles_path=cell_rles_path,
            nucleus_start_idx=nucleus_range[0],
            nucleus_end_idx=nucleus_range[1],
            cell_start_idx=cell_range[0],
            cell_end_idx=cell_range[1],
            max_vertices=args.max_vertices,
            simplify_epsilon=args.simplify_epsilon,
            out_nucleus_vertices_path=paths["nucleus_vertices"],
            out_nucleus_num_vertices_path=paths["nucleus_num_vertices"],
            out_nucleus_int_id_path=paths["nucleus_int_id"],
            out_nucleus_pyramid_path=paths["nucleus_pyramid"],
            out_cell_vertices_path=paths["cell_vertices"],
            out_cell_num_vertices_path=paths["cell_num_vertices"],
            out_cell_int_id_path=paths["cell_int_id"],
            out_cell_pyramid_path=paths["cell_pyramid"],
            out_bboxes_path=paths["bboxes"],
            out_z_levels_path=paths["z_levels"],
            disable_build_pyramids=False,
        )
        for name, path in paths.items():
            output_lists[name].append(path)
        print(
            f"[polygons] {chunk_index + 1}/{len(cell_ranges)} "
            f"cells={cell_range[0]}:{cell_range[1]}",
            flush=True,
        )

    dataset = stage.join(
        nucleus_rles_path=str(nucleus_rles_path),
        cell_rles_path=cell_rles_path,
        nucleus_vertices_path_list=output_lists["nucleus_vertices"],
        nucleus_num_vert_path_list=output_lists["nucleus_num_vertices"],
        nucleus_int_id_path_list=output_lists["nucleus_int_id"],
        nucleus_pyramid_path_list=output_lists["nucleus_pyramid"],
        cell_vertices_path_list=output_lists["cell_vertices"],
        cell_num_vert_path_list=output_lists["cell_num_vertices"],
        cell_int_id_path_list=output_lists["cell_int_id"],
        cell_pyramid_path_list=output_lists["cell_pyramid"],
        bboxes_path_list=output_lists["bboxes"],
        z_level_path_list=output_lists["z_levels"],
        full_img_height=full_height,
        full_img_width=full_width,
        max_vertices=args.max_vertices,
        num_boundary_cell=num_cells,
        num_expanded_cell=0,
        boundary_analyte_name=args.boundary_analyte,
        interior_analyte_name="None",
        expansion_distance_um=0.0,
        large_cell_scale_factor=args.large_cell_scale,
        pixel_to_physical_transform=pixel_to_physical,
        num_thread=args.threads,
        disable_build_pyramids=False,
    )
    dataset.save_to_zarr_location(str(out_path))

    summary = {
        "cells_zarr_zip": str(out_path),
        "num_cells": num_cells,
        "cell_rles": cell_rles_path,
        "base_shape": [full_height, full_width],
        "pixel_to_physical_transform": pixel_to_physical.tolist(),
        "contains_nuclei": False,
        "segmentation_scope": "external large-cell branch only",
        "dataset_builder": "production GENERATE_CELLS_DATASET functions",
    }
    summary_path = out_path.with_suffix(out_path.suffix + ".summary.json")
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[done] cells: {out_path}")
    print(f"[done] summary: {summary_path}")


if __name__ == "__main__":
    main()
