# Turing bridge scripts

These four scripts let an external PyTorch Ridgepath model use Turing's existing preprocessing,
decoder, metrics, merging, and Zarr code. They add standalone entry points; they do not replace or
modify Turing core files.

They were tested with Turing revision:

```text
0e7e8d8b325187852e52bcd86edfdde785d47d2c
```

## Install

From the `pytorch_seg_delivery` directory, copy the Python files into a checkout of Turing:

```bash
TURING_REPO=/path/to/turing

cp add_to_turing/*.py \
  "${TURING_REPO}/lib/python/turing/analysis/cell_segmentation/"

cd "${TURING_REPO}"
bazel build //:devpipes_env
```

No `BUILD.bazel` change is required. Run each script through Turing's environment:

```bash
bazel-bin/devpipes_env.sh python \
  lib/python/turing/analysis/cell_segmentation/<script>.py --help
```

## Workflows

- Patch evaluation: `eval_external_logits.py`
- Whole-image preparation: `prepare_external_largecell_tiles.py`
- Whole-image large-cell decoding and merging: `decode_external_largecell_production.py`
- Final large-cell-only Zarr creation: `build_external_largecell_zarr_production.py`

The whole-image sequence is preparation, PyTorch inference in `pytorch_seg_delivery`, Turing
decoding, and Zarr creation. Use the same Turing revision, model `config.json` and `best.pth`, input
data, and command parameters for reproducible results.
