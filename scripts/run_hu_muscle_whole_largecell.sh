#!/usr/bin/env bash
set -euo pipefail

PYTORCH_CODE_REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTORCH_ARTIFACT_REPO=${PYTORCH_SEG_ARTIFACT_ROOT:-${PYTORCH_CODE_REPO}}
TURING_REPO=${PYTORCH_SEG_TURING_REPO:-/mnt/home/ruizhi.yuan/turing}
TORCH_PYTHON=${PYTORCH_SEG_PYTHON:-/mnt/home/ruizhi.yuan/miniforge/miniforge3/envs/torch_ssl/bin/python}

SAMPLE_DIR=${PYTORCH_SEG_SAMPLE_DIR:-/mnt/home/fangda.li/yard/turing_exp/sample_images_and_features/hu_muscle_cs_20240405}
MORPHOLOGY_PATH=${SAMPLE_DIR}/morphology.ome.tif
RUN_DIR=${PYTORCH_SEG_RUN_DIR:-${PYTORCH_ARTIFACT_REPO}/runs/convnext_dino_hires_muscle}

# Keep the prepared inputs, large temporary logits, and final Ziggy artifact on
# shared deck storage. The four stages below preserve Turing's production CPU
# large-cell behavior while replacing only the network forward pass.
OUTPUT_DIR=${PYTORCH_SEG_OUTPUT_DIR:-/mnt/deck/2/ruizhi.yuan/turing_exp/hu_muscle_cs_20240405_dinov3_largecell}
PREP_DIR=${OUTPUT_DIR}/turing_production_inputs
LOGITS_DIR=${OUTPUT_DIR}/pytorch_production_logits
DECODE_DIR=${OUTPUT_DIR}/turing_production_decode
CELLS_PATH=${OUTPUT_DIR}/cells_dinov3_largecell_production.zarr.zip

mkdir -p "${OUTPUT_DIR}"

if [[ -s "${PREP_DIR}/tiles.csv" && -s "${PREP_DIR}/metadata.json" ]]; then
  echo "[reuse] completed Turing production inputs: ${PREP_DIR}"
else
  cd "${TURING_REPO}"
  CUDA_VISIBLE_DEVICES=-1 bazel-bin/devpipes_env.sh python \
    lib/python/turing/analysis/cell_segmentation/prepare_external_largecell_tiles.py \
    --morphology-focus-multifile "${SAMPLE_DIR}/morphology_focus_multifile" \
    --boundary-channel-name "ATP1A1/CD45/E-Cadherin" \
    --out "${PREP_DIR}" \
    --segmentor-scale 0.25
fi

cd "${PYTORCH_CODE_REPO}"
CUDA_VISIBLE_DEVICES=0 "${TORCH_PYTHON}" scripts/infer_prepared_largecell_tiles.py \
  --manifest "${PREP_DIR}/tiles.csv" \
  --metadata "${PREP_DIR}/metadata.json" \
  --config "${RUN_DIR}/config.json" \
  --checkpoint "${RUN_DIR}/best.pth" \
  --out "${LOGITS_DIR}" \
  --device cuda \
  --amp-dtype none

cd "${TURING_REPO}"
CUDA_VISIBLE_DEVICES=-1 bazel-bin/devpipes_env.sh python \
  lib/python/turing/analysis/cell_segmentation/decode_external_largecell_production.py \
  --manifest "${LOGITS_DIR}/tiles.csv" \
  --preparation-metadata "${PREP_DIR}/metadata.json" \
  --out "${DECODE_DIR}" \
  --threads 8

CUDA_VISIBLE_DEVICES=-1 bazel-bin/devpipes_env.sh python \
  lib/python/turing/analysis/cell_segmentation/build_external_largecell_zarr_production.py \
  --cell-rles "${DECODE_DIR}/cell_rles.pb" \
  --morphology "${MORPHOLOGY_PATH}" \
  --out "${CELLS_PATH}" \
  --work-dir "${DECODE_DIR}/zarr_work" \
  --boundary-analyte "ATP1A1/CD45/E-Cadherin" \
  --large-cell-scale 0.25 \
  --threads 8

echo "Large-cell dataset: ${CELLS_PATH}"
echo "Ziggy: https://ziggy.txgmesh.net/?image=yard%2Ffangda.li%2Fturing_exp%2Fsample_images_and_features%2Fhu_muscle_cs_20240405%2Fmorphology_focus_multifile%2Fmorphology_focus_0000.ome.tif&cell_seg=deck%2F2%2Fruizhi.yuan%2Fturing_exp%2Fhu_muscle_cs_20240405_dinov3_largecell%2Fcells_dinov3_largecell_production.zarr.zip&layers=cell~image&cell_v=outlined"
