#!/usr/bin/env python
"""Production-equivalent Turing decode for external large-cell Ridgepath logits.

The only replaced operation is the neural network that produced each raw
nine-channel logit tensor.  This script matches ``PathSegmenter.run``,
``segment_cells``, and the ``SEGMENT_CELLS`` join behavior after inference.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf
from tensorflow.core.framework import graph_pb2  # pylint: disable=no-name-in-module

from turing.engine.cell_segmentation import cell_segmentation as cs_mod
from turing.engine.cell_segmentation.cell_segmentation import boundary_instance_label
from turing.engine.cell_segmentation.cell_segmentation_structs import POST_MODEL_NAME
from turing.engine.cell_segmentation.cellseg_o3 import (
    label_expansion,
    merge_saved_rles,
    offset_rles,
    path_segmenter_post_process,
    save_rles_to_pb,
)
from turing.engine.cell_segmentation.segmentation_methods import SegmentationPriority
from turing.engine.cell_segmentation.stages.segment_cells import (
    BOUNDARY_SEG_PARAM,
    cell_label_to_rle_list,
)


def load_post_model():
    post_graph = graph_pb2.GraphDef()
    post_path = os.path.join(os.path.dirname(cs_mod.__file__), POST_MODEL_NAME)
    with open(post_path, "rb") as handle:
        post_graph.ParseFromString(handle.read())

    @tf.function(input_signature=[tf.TensorSpec([1, None, None, 9], tf.float32)])
    def run_post(model_output):
        direct_map, stationary_px = tf.graph_util.import_graph_def(
            post_graph,
            name="",
            input_map={"model_output": model_output},
            return_elements=["Identity:0", "Identity_1:0"],
        )
        return direct_map, stationary_px

    return run_post


def decode_target_labels(run_post, logits: np.ndarray) -> tuple[np.ndarray, dict[int, float]]:
    if logits.ndim != 3 or logits.shape[-1] != 9:
        raise ValueError(f"expected [H,W,9] logits, found {logits.shape}")
    direct_map, stationary_px = run_post(
        tf.convert_to_tensor(logits[np.newaxis], dtype=tf.float32)
    )
    direct = direct_map.numpy()[:, :, :, 0].astype(np.int8)
    stationary = stationary_px.numpy()[:, :, :, 0].astype(np.int8)
    labels, scores = path_segmenter_post_process(
        direct,
        stationary,
        merge_threshold=BOUNDARY_SEG_PARAM.merge_threshold,
        merge_center_radius=BOUNDARY_SEG_PARAM.merge_radius,
        min_support=BOUNDARY_SEG_PARAM.min_support,
        correct_dict=BOUNDARY_SEG_PARAM.correct_dict,
        filter_by_alignment=BOUNDARY_SEG_PARAM.filter_by_alignment,
        inner_threshold=BOUNDARY_SEG_PARAM.inner_threshold,
        gradient_threshold=BOUNDARY_SEG_PARAM.gradient_threshold,
        distance_threshold=BOUNDARY_SEG_PARAM.distance_threshold,
        num_threads=1,
    )
    return np.asarray(labels).astype(np.int32, copy=False), scores


def resize_labels_like_path_segmenter(
    labels: np.ndarray, raw_height: int, raw_width: int
) -> np.ndarray:
    if labels.shape == (raw_height, raw_width):
        return labels
    return tf.image.resize(
        labels[np.newaxis, :, :, np.newaxis],
        [raw_height, raw_width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        name=None,
    ).numpy()[0, :, :, 0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode external logits with the production Turing large-cell path"
    )
    parser.add_argument("--manifest", required=True, help="PyTorch logits tiles.csv")
    parser.add_argument("--preparation-metadata", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    metadata_path = Path(args.preparation_metadata).resolve()
    out_dir = Path(args.out).resolve()
    chunk_dir = out_dir / "production_chunk_outputs"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]
    if not rows:
        raise ValueError("logits manifest contains no rows")
    with metadata_path.open() as handle:
        metadata = json.load(handle)
    if metadata.get("pipeline") != "turing_cpu_segment_cells_large_cell_pre_inference":
        raise ValueError("preparation metadata is not from the production tile preparer")

    full_height = int(metadata["full_height"])
    full_width = int(metadata["full_width"])
    min_area_px = int(metadata["min_area_px"])
    max_area_px = int(metadata["max_area_px"])
    expansion_dist = float(metadata["expansion_dist"])
    alignment_threshold = float(metadata["alignment_threshold"])
    num_outer_chunks = int(metadata["num_outer_chunks"])

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    run_post = load_post_model()

    rles_by_chunk: dict[int, list] = defaultdict(list)
    scores_by_chunk: dict[int, list[float]] = defaultdict(list)
    raw_instance_count = 0
    for row_index, row in enumerate(rows):
        logits = np.load(row["logits_path"], mmap_mode="r")
        labels, score_mapping = decode_target_labels(run_post, np.asarray(logits))
        target_shape = (int(row["target_height"]), int(row["target_width"]))
        if labels.shape != target_shape:
            raise ValueError(
                f"tile {row['tile_id']} decoded shape {labels.shape} != {target_shape}"
            )
        raw_height = int(row["raw_height"])
        raw_width = int(row["raw_width"])
        labels = resize_labels_like_path_segmenter(labels, raw_height, raw_width)
        labels = label_expansion(
            labels.astype(np.uint32), expansion_dist=expansion_dist
        )

        x = int(row["x"])
        y = int(row["y"])
        overlap = int(row["inner_overlap"])
        exclusion_sides = (
            int(y > 0),
            int(y + raw_height < full_height),
            int(x > 0),
            int(x + raw_width < full_width),
        )
        edge_labels = boundary_instance_label(
            labels, exclusion_sides, overlap // 2 if overlap else 0
        )
        low_score_labels = np.asarray(
            [
                label
                for label, score in score_mapping.items()
                if score < alignment_threshold
            ],
            dtype=np.uint32,
        )
        valid_score_instances = np.setdiff1d(
            list(score_mapping.keys()), edge_labels, assume_unique=True
        )
        chunk_id = int(row["chunk_id"])
        scores_by_chunk[chunk_id].extend(
            float(score_mapping[int(label)]) for label in valid_score_instances
        )
        excluded = np.union1d(low_score_labels, edge_labels)
        tile_rles = cell_label_to_rle_list(
            labels,
            excluded,
            min_area_px=min_area_px,
            max_area_px=max_area_px,
            rle_priority=SegmentationPriority.BOUNDARY_LARGE_CELL,
            score_mapping=score_mapping,
        )
        shifted_rles = offset_rles(
            tile_rles,
            full_height,
            full_width,
            offset_height=y,
            offset_width=x,
        )
        rles_by_chunk[chunk_id].extend(shifted_rles)
        raw_instance_count += len(shifted_rles)
        print(
            f"[decode] {row_index + 1}/{len(rows)} tiles; "
            f"tile_instances={len(shifted_rles)} raw_total={raw_instance_count}",
            flush=True,
        )

    chunk_rle_paths: list[str] = []
    chunk_score_paths: list[str] = []
    for chunk_id in range(num_outer_chunks):
        rle_path = chunk_dir / f"chunk_{chunk_id:04d}_rles.pb"
        score_path = chunk_dir / f"chunk_{chunk_id:04d}_scores.npy"
        save_rles_to_pb(rles_by_chunk[chunk_id], str(rle_path))
        np.save(score_path, np.asarray(scores_by_chunk[chunk_id], dtype=np.float64))
        chunk_rle_paths.append(str(rle_path))
        chunk_score_paths.append(str(score_path))

    merged_rles_path = out_dir / "cell_rles.pb"
    num_instances = merge_saved_rles(
        chunk_rle_paths,
        merge_by_label=False,
        relabel_rles=True,
        remove_fragments=True,
        output_path=str(merged_rles_path),
        overlap_threshold=0.1,
        threads=args.threads,
    )
    chunk_scores = [np.load(path) for path in chunk_score_paths]
    all_scores = np.concatenate(chunk_scores) if chunk_scores else np.array([])
    scores_path = out_dir / "seg_scores.npz"
    np.savez(scores_path, all_scores)

    summary = {
        "pipeline": "turing_cpu_segment_cells_large_cell_post_inference",
        "external_operation": "neural_network_forward_only",
        "num_tiles": len(rows),
        "num_outer_chunks": num_outer_chunks,
        "raw_tile_instances": raw_instance_count,
        "merged_instances": int(num_instances),
        "full_shape": [full_height, full_width],
        "cell_rles": str(merged_rles_path),
        "seg_scores": str(scores_path),
        "production_parameters": {
            "merge_threshold": BOUNDARY_SEG_PARAM.merge_threshold,
            "merge_radius": BOUNDARY_SEG_PARAM.merge_radius,
            "min_support": BOUNDARY_SEG_PARAM.min_support,
            "correct_dict": BOUNDARY_SEG_PARAM.correct_dict,
            "filter_by_alignment": BOUNDARY_SEG_PARAM.filter_by_alignment,
            "inner_threshold": BOUNDARY_SEG_PARAM.inner_threshold,
            "gradient_threshold": BOUNDARY_SEG_PARAM.gradient_threshold,
            "distance_threshold": BOUNDARY_SEG_PARAM.distance_threshold,
            "alignment_threshold": alignment_threshold,
            "min_area_px": min_area_px,
            "max_area_px": max_area_px,
            "expansion_dist": expansion_dist,
            "chunk_merge_overlap_threshold": 0.1,
        },
    }
    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[done] merged instances: {num_instances}")
    print(f"[done] RLEs: {merged_rles_path}")
    print(f"[done] scores: {scores_path}")


if __name__ == "__main__":
    main()
