"""DINO checkpoint restore into the ResNet18 segmentation encoder.

The DINO student/teacher state_dict stores the backbone under ``module.backbone.*`` with a
4-channel ``conv1.weight`` (64,4,7,7). The seg model uses a 2-channel conv1, so we transfer
weights by **marker NAME** (not a raw index), slicing the DINO conv1 input channels to the
seg image's channel order.

Channel facts (provided by the user; NO ssl_4ch prep script exists in the repo to auto-verify,
so the order is config-driven and printed for inspection):
  * SSL 4-ch order  : ['DAPI', 'boundary', '18S', 'avim']   (DINO conv1 input channels)
  * large_cell_boundary seg image order: ['boundary', 'DAPI']  (ch0=boundary, ch1=DAPI)
  => seg pos 0 (boundary) <- DINO idx 1 ; seg pos 1 (DAPI) <- DINO idx 0  => slice [1, 0].

Mismatches are never silently ignored: shape mismatches raise, and loaded/missing/unexpected
keys are printed.
"""
import torch

BACKBONE_PREFIX = "module.backbone."


def load_dino_backbone_state_dict(ckpt_path, which="student"):
    """Return (backbone_state_dict, epoch) with any wrapper prefix before ``backbone.`` stripped.

    DDP-wrapped modules store the backbone as ``module.backbone.*``; a non-DDP module (e.g. the
    GroupNorm iBOT teacher, which is not DDP-wrapped because it has no BatchNorm buffers to sync)
    stores it as plain ``backbone.*``. Strip an optional leading ``module.`` first, then ``backbone.``.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if which not in ck:
        raise KeyError(f"'{which}' not in checkpoint (keys: {list(ck.keys())})")
    sd = ck[which]
    bb = {}
    for k, v in sd.items():
        kk = k[len("module."):] if k.startswith("module.") else k
        if kk.startswith("backbone."):
            bb[kk[len("backbone."):]] = v
    if not bb:
        raise ValueError(
            f"no 'backbone.*' (or 'module.backbone.*') keys in checkpoint['{which}'] "
            f"(sample keys: {list(sd)[:5]})")
    return bb, ck.get("epoch")


def resolve_conv1_index_map(seg_channels, ssl_channels):
    """seg image channel order -> list of DINO conv1 input indices (by marker name)."""
    idx = []
    for pos, name in enumerate(seg_channels):
        if name not in ssl_channels:
            raise ValueError(
                f"seg channel '{name}' (pos {pos}) not in SSL channels {ssl_channels}"
            )
        idx.append(ssl_channels.index(name))
    return idx


def restore_dino_into_resnet18(seg_model, ckpt_path, seg_channels, ssl_channels,
                               which="student", verbose=True):
    """Load DINO backbone into ``seg_model.encoder.backbone`` (torchvision ResNet18).

    Slices the 4-ch DINO conv1 down to the seg channel order by marker name. Raises on any
    shape mismatch; prints loaded/missing/unexpected key counts.
    """
    if not hasattr(seg_model.encoder, "backbone"):
        raise TypeError("DINO restore requires the resnet18 encoder (encoder.backbone missing). "
                        "tenxnet_small cannot be DINO-initialized.")
    target = seg_model.encoder.backbone
    target_sd = target.state_dict()

    bb, epoch = load_dino_backbone_state_dict(ckpt_path, which)

    # --- conv1 channel-name slicing (verify before slicing) ---
    if "conv1.weight" not in bb:
        raise KeyError("DINO backbone has no conv1.weight")
    dino_conv1 = bb["conv1.weight"]
    if dino_conv1.ndim != 4 or dino_conv1.shape[1] != len(ssl_channels):
        raise ValueError(
            f"DINO conv1 shape {tuple(dino_conv1.shape)} incompatible with ssl_channels "
            f"{ssl_channels} (expected in_channels={len(ssl_channels)})")
    cmap = resolve_conv1_index_map(seg_channels, ssl_channels)
    sliced = dino_conv1[:, cmap, :, :].clone()
    tgt_conv1 = target_sd["conv1.weight"]
    if sliced.shape != tgt_conv1.shape:
        raise ValueError(f"sliced conv1 {tuple(sliced.shape)} != seg conv1 {tuple(tgt_conv1.shape)}")
    bb = dict(bb)
    bb["conv1.weight"] = sliced

    # --- explicit key reconciliation (no silent mismatches) ---
    tkeys, pkeys = set(target_sd), set(bb)
    shape_mismatch = sorted(k for k in tkeys & pkeys if target_sd[k].shape != bb[k].shape)
    if shape_mismatch:
        details = [f"{k}: seg{tuple(target_sd[k].shape)} vs dino{tuple(bb[k].shape)}"
                   for k in shape_mismatch]
        raise ValueError("shape mismatch on keys:\n  " + "\n  ".join(details))
    loaded = sorted(tkeys & pkeys)
    missing = sorted(tkeys - pkeys)      # in seg model, absent from DINO (kept at init)
    unexpected = sorted(pkeys - tkeys)   # in DINO, absent from seg model (ignored)

    result = target.load_state_dict(bb, strict=False)
    # sanity: strict=False should agree with our manual reconciliation
    assert set(result.missing_keys) == set(missing)
    assert set(result.unexpected_keys) == set(unexpected)

    if verbose:
        print(f"[restore] DINO '{which}' backbone from {ckpt_path} (epoch {epoch})")
        print(f"[restore] conv1 channel map (seg_pos -> marker -> dino_idx): "
              + ", ".join(f"{p}->{m}->{cmap[p]}" for p, m in enumerate(seg_channels)))
        print(f"[restore] loaded={len(loaded)} missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"[restore]   MISSING (kept at init): {missing}")
        if unexpected:
            print(f"[restore]   UNEXPECTED (ignored):   {unexpected}")
    return dict(loaded=loaded, missing=missing, unexpected=unexpected, epoch=epoch, conv1_map=cmap)


# Input conv of the tenxnet stem, by stem_type:
#   v1 (tenxnet_recipe): stem[0] = (Conv-BN-ReLU) -> "stem.0.0.weight"
#   v0 (tenxnet_small):  stem    = (Conv-BN-ReLU) -> "stem.0.weight"
TENXNET_STEM_CONV_CANDIDATES = ("stem.0.0.weight", "stem.0.weight")


def _detect_tenxnet_stem_conv(target_sd):
    """Return the stem input-conv key present in the seg encoder (v1 'stem.0.0.weight' / v0 'stem.0.weight')."""
    for k in TENXNET_STEM_CONV_CANDIDATES:
        if k in target_sd:
            return k
    raise KeyError(f"no tenxnet stem input conv in encoder (tried {TENXNET_STEM_CONV_CANDIDATES})")


def restore_dino_into_tenxnet(seg_model, ckpt_path, seg_channels, ssl_channels,
                              which="student", verbose=True):
    """Load a tenxnet DINO backbone into ``seg_model.encoder`` (a ``TenxnetSmallResNet``).

    Handles both ``tenxnet_recipe`` (v1 stem) and ``tenxnet_small`` (v0 stem): the seg encoder *is*
    the trunk (no ``.backbone`` wrapper), and the DINO backbone (``ssl_dino/tenxnet_backbone.py``)
    shares its exact param names, so stripping ``module.backbone.`` lands straight on ``encoder.*``.
    The stem input conv is auto-detected (v1 ``stem.0.0.weight`` / v0 ``stem.0.weight``) and sliced
    to the seg channel order by marker name -- identity for matching 2-ch ``[boundary, DAPI]``. The
    DINOHead is not part of the trunk and is not loaded. Raises on any shape mismatch.
    """
    if hasattr(seg_model.encoder, "backbone"):
        raise TypeError("tenxnet restore expects model.encoder (no .backbone); got a "
                        "resnet18-style encoder -- use restore_dino_into_resnet18 instead.")
    target = seg_model.encoder
    target_sd = target.state_dict()
    stem_conv = _detect_tenxnet_stem_conv(target_sd)  # v0 or v1, from the seg encoder

    bb, epoch = load_dino_backbone_state_dict(ckpt_path, which)

    # --- stem input-conv channel-name slicing (verify before slicing) ---
    if stem_conv not in bb:
        raise KeyError(f"DINO backbone has no {stem_conv} (seg encoder expects it)")
    dino_conv = bb[stem_conv]
    if dino_conv.ndim != 4 or dino_conv.shape[1] != len(ssl_channels):
        raise ValueError(
            f"DINO {stem_conv} shape {tuple(dino_conv.shape)} incompatible with "
            f"ssl_channels {ssl_channels} (expected in_channels={len(ssl_channels)})")
    cmap = resolve_conv1_index_map(seg_channels, ssl_channels)
    sliced = dino_conv[:, cmap, :, :].clone()
    tgt_conv = target_sd[stem_conv]
    if sliced.shape != tgt_conv.shape:
        raise ValueError(
            f"sliced {stem_conv} {tuple(sliced.shape)} != seg {tuple(tgt_conv.shape)}")
    bb = dict(bb)
    bb[stem_conv] = sliced

    # --- explicit key reconciliation (no silent mismatches) ---
    tkeys, pkeys = set(target_sd), set(bb)
    shape_mismatch = sorted(k for k in tkeys & pkeys if target_sd[k].shape != bb[k].shape)
    if shape_mismatch:
        details = [f"{k}: seg{tuple(target_sd[k].shape)} vs dino{tuple(bb[k].shape)}"
                   for k in shape_mismatch]
        raise ValueError("shape mismatch on keys:\n  " + "\n  ".join(details))
    loaded = sorted(tkeys & pkeys)
    missing = sorted(tkeys - pkeys)      # in seg encoder, absent from DINO (kept at init)
    unexpected = sorted(pkeys - tkeys)   # in DINO, absent from seg encoder (ignored)

    result = target.load_state_dict(bb, strict=False)
    assert set(result.missing_keys) == set(missing)
    assert set(result.unexpected_keys) == set(unexpected)

    if verbose:
        print(f"[restore] DINO '{which}' tenxnet trunk from {ckpt_path} (epoch {epoch}) "
              f"[stem conv {stem_conv}]")
        print(f"[restore] stem conv channel map (seg_pos -> marker -> dino_idx): "
              + ", ".join(f"{p}->{m}->{cmap[p]}" for p, m in enumerate(seg_channels)))
        print(f"[restore] loaded={len(loaded)} missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"[restore]   MISSING (kept at init): {missing}")
        if unexpected:
            print(f"[restore]   UNEXPECTED (ignored):   {unexpected}")
    return dict(loaded=loaded, missing=missing, unexpected=unexpected, epoch=epoch, conv1_map=cmap)


# Back-compat alias: the recipe path is just the generic tenxnet restore (auto-detects the stem).
restore_dino_into_tenxnet_recipe = restore_dino_into_tenxnet


def restore_seg_encoder(seg_model, ckpt_path, verbose=True):
    """Load ONLY the encoder subtree from a pytorch_seg checkpoint into ``seg_model.encoder``.

    Used by the ``init: seg_encoder`` control experiment: take a train-from-scratch seg checkpoint,
    keep its (trained) encoder, and DISCARD its decoder + output heads (those stay at random init in
    ``seg_model`` -- we simply don't load them). The source checkpoint stores a full seg model under
    ``ckpt["model"]`` with every encoder param prefixed ``encoder.``; we filter to those keys, strip
    the prefix, and load into ``seg_model.encoder``.

    The source and target encoder MUST be the same architecture + norm (e.g. both tenxnet_recipe BN)
    so channels/keys align 1:1 -- no slicing, and we assert **0 missing encoder keys** (a partial
    match would silently leave part of the encoder random, defeating the control). Raises otherwise.
    """
    target = seg_model.encoder
    target_sd = target.state_dict()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" not in ck:
        raise KeyError(f"checkpoint has no 'model' key (keys: {list(ck.keys())})")
    full_sd = ck["model"]
    enc = {k[len("encoder."):]: v for k, v in full_sd.items() if k.startswith("encoder.")}
    if not enc:
        raise ValueError(f"no 'encoder.*' keys in checkpoint['model'] (sample: {list(full_sd)[:5]})")

    tkeys, pkeys = set(target_sd), set(enc)
    shape_mismatch = sorted(k for k in tkeys & pkeys if target_sd[k].shape != enc[k].shape)
    if shape_mismatch:
        details = [f"{k}: seg{tuple(target_sd[k].shape)} vs ckpt{tuple(enc[k].shape)}"
                   for k in shape_mismatch]
        raise ValueError("encoder shape mismatch on keys:\n  " + "\n  ".join(details))
    missing = sorted(tkeys - pkeys)      # encoder keys absent from source ckpt -> would stay random
    unexpected = sorted(pkeys - tkeys)   # source keys absent from target encoder
    if missing:
        raise ValueError(
            f"source checkpoint is missing {len(missing)} encoder keys "
            f"(would stay at random init, corrupting the frozen-encoder control): {missing[:8]}"
            + (" ..." if len(missing) > 8 else "")
            + "\n=> the source run's encoder must be the SAME architecture+norm as this config "
              "(e.g. both tenxnet_recipe encoder_norm=bn).")

    result = target.load_state_dict(enc, strict=False)
    assert not result.missing_keys, result.missing_keys
    assert set(result.unexpected_keys) == set(unexpected)

    if verbose:
        print(f"[restore] seg ENCODER-ONLY from {ckpt_path} (source epoch {ck.get('epoch','?')})")
        print(f"[restore] loaded={len(tkeys & pkeys)} encoder tensors; "
              f"decoder + heads left at RANDOM init (discarded from source)")
        if unexpected:
            print(f"[restore]   UNEXPECTED source encoder keys (ignored): {unexpected}")
    return dict(loaded=sorted(tkeys & pkeys), unexpected=unexpected, epoch=ck.get("epoch"))


def restore_seg_full(seg_model, ckpt_path, verbose=True):
    """Load the ENTIRE model (encoder + decoder + heads) from a pytorch_seg checkpoint into ``seg_model``.

    Used by the ``init: seg_full`` LP-FT warm-start: take a completed frozen-encoder ("linear-probe")
    run and continue with the encoder UNFROZEN, but with a FRESH optimizer / schedule / small encoder LR
    (which a plain ``--resume`` cannot give -- resume restores the saved optimizer's param-groups, so the
    encoder would inherit the probe's hot base_lr and overwrite the pretrained features). Here we load
    weights only; train_seg builds the optimizer from the NEW config.

    Source and target MUST be the same architecture (same encoder/decoder/head config) so keys align
    1:1; we assert 0 missing and 0 unexpected keys (a partial load would silently leave part of the
    model at random init). Build the target with ``model_kwargs.pretrained: false`` -- the ConvNeXt
    weights come from this checkpoint (identical to DINOv3 since the probe kept the encoder frozen), so
    no HF download is needed.
    """
    raw = seg_model.module if hasattr(seg_model, "module") else seg_model
    target_sd = raw.state_dict()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" not in ck:
        raise KeyError(f"checkpoint has no 'model' key (keys: {list(ck.keys())})")
    src = ck["model"]

    tkeys, pkeys = set(target_sd), set(src)
    shape_mismatch = sorted(k for k in tkeys & pkeys if target_sd[k].shape != src[k].shape)
    if shape_mismatch:
        details = [f"{k}: seg{tuple(target_sd[k].shape)} vs ckpt{tuple(src[k].shape)}"
                   for k in shape_mismatch]
        raise ValueError("model shape mismatch on keys:\n  " + "\n  ".join(details))
    missing = sorted(tkeys - pkeys)      # target params absent from source -> would stay random
    unexpected = sorted(pkeys - tkeys)   # source params absent from target
    if missing or unexpected:
        raise ValueError(
            f"seg_full warm-start key mismatch: {len(missing)} missing, {len(unexpected)} unexpected.\n"
            f"  missing (would stay random): {missing[:8]}{' ...' if len(missing) > 8 else ''}\n"
            f"  unexpected (in ckpt, not model): {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}\n"
            "=> the source run must be the SAME architecture/config as this one.")

    raw.load_state_dict(src, strict=True)
    if verbose:
        print(f"[restore] seg FULL model from {ckpt_path} (source epoch {ck.get('epoch','?')}) "
              f"-> {len(tkeys)} tensors loaded (encoder + decoder + heads); fresh optimizer/schedule")
    return dict(loaded=sorted(tkeys), epoch=ck.get("epoch"))
