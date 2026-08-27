"""Swappable feature-pyramid encoders for the ported seg model.

Both expose ``forward_features(x) -> {"2":C2,"3":C3,"4":C4,"5":C5}`` (NCHW) and an
``output_specs`` dict ({level: channels}) so the FPN is built backbone-agnostically.

  * ``resnet18``  -- the DINO drop-in: torchvision ResNet18 with conv1 -> in_chans, tapping
    layer1..layer4 (C2-C5 = 64/128/256/512 @ stride 4/8/16/32). Identical construction to
    ssl_dino/resnet4ch.py:build_resnet18_4ch, so the DINO checkpoint loads with a plain
    state_dict copy (see restore.py). Stem stride-2.
  * ``tenxnet_small`` -- faithful port of tenxnet/vision/models/encoders/resnet.py (ResNet
    class, stem_type v0): 7x7 stride-1 stem + 3x3 stride-2 maxpool, then 4 basic-block groups
    (start_filter 16, num_filters [16,32,64,128], block_repeats [2,2,2,2]); endpoints "2".."5"
    at stride 2/4/8/16, channels 16/32/64/128. Basic block mirrors nn_blocks.ResidualBlock
    (projection conv1x1+BN on the first block of each group). SE / resnetd / stochastic-depth
    are unused in the ridgepath config and omitted.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from .layers import bn, make_gn


# --------------------------------------------------------------------------- resnet18 drop-in
def _build_resnet18(in_chans: int):
    """Mirror of ssl_dino/resnet4ch.py:build_resnet18_4ch (conv1 -> in_chans, else stock)."""
    model = tv_models.resnet18(weights=None)
    old = model.conv1  # Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
    model.conv1 = nn.Conv2d(in_chans, old.out_channels, kernel_size=old.kernel_size,
                            stride=old.stride, padding=old.padding, bias=(old.bias is not None))
    return model


class ResNet18Features(nn.Module):
    def __init__(self, in_chans: int = 2):
        super().__init__()
        self.backbone = _build_resnet18(in_chans)
        # avgpool/fc are unused by feature extraction; drop fc so it carries no dead params
        # (mirrors how DINO's MultiCropWrapper sets fc=Identity).
        self.backbone.fc = nn.Identity()
        self.output_specs = {"2": 64, "3": 128, "4": 256, "5": 512}

    def forward_features(self, x):
        b = self.backbone
        x = b.relu(b.bn1(b.conv1(x)))
        x = b.maxpool(x)
        c2 = b.layer1(x)
        c3 = b.layer2(c2)
        c4 = b.layer3(c3)
        c5 = b.layer4(c4)
        return {"2": c2, "3": c3, "4": c4, "5": c5}

    forward = forward_features


# --------------------------------------------------------------------------- faithful tenxnet small
class DropPath(nn.Module):
    """Stochastic depth (== nn_layers.StochasticDepth): drop the residual branch per-sample
    with prob ``drop_rate`` during training, scaling kept samples by 1/keep_prob."""

    def __init__(self, drop_rate: float = 0.0):
        super().__init__()
        self.drop_rate = float(drop_rate)

    def forward(self, x):
        if self.drop_rate == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_rate
        mask = x.new_empty((x.shape[0], 1, 1, 1)).bernoulli_(keep)
        return x / keep * mask


class BasicBlock(nn.Module):
    """== nn_blocks.ResidualBlock (no SE / resnetd). Optional stochastic depth on the residual."""

    def __init__(self, in_ch: int, filters: int, stride: int, use_projection: bool,
                 drop_path_rate: float = 0.0, norm_layer=None):
        super().__init__()
        nl = norm_layer or bn  # None -> BatchNorm (default path unchanged)
        self.use_projection = use_projection
        if use_projection:
            self.shortcut = nn.Conv2d(in_ch, filters, kernel_size=1, stride=stride, bias=False)
            self.norm0 = nl(filters)
        self.conv1 = nn.Conv2d(in_ch, filters, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nl(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nl(filters)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x):
        shortcut = x
        if self.use_projection:
            shortcut = self.norm0(self.shortcut(x))
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.drop_path(out)  # stochastic depth on residual, before add (matches tenxnet)
        return F.relu(out + shortcut)


class TenxnetSmallResNet(nn.Module):
    """Faithful tenxnet ResNet (encoders/resnet.py).

    ``stem_type`` v0 = single 7x7 stride-1 conv; v1 = three 3x3 stride-1 convs. ``expose_stem``
    adds endpoint "1" (stem output, full res). ``stochastic_depth_rate`` matches tenxnet's
    per-group schedule rate = init * (i+2) / (num_lvls+1) (same rate for all blocks in a group).
    """

    def __init__(self, in_chans: int = 2, start_filter: int = 16,
                 num_filters=(16, 32, 64, 128), block_repeats=(2, 2, 2, 2),
                 stem_type: str = "v0", stochastic_depth_rate: float = 0.0,
                 expose_stem: bool = False, norm_layer=None):
        super().__init__()
        self.expose_stem = expose_stem
        nl = norm_layer or bn  # None -> BatchNorm (default path unchanged)

        def cbr(cin, k):
            return nn.Sequential(
                nn.Conv2d(cin, start_filter, kernel_size=k, stride=1, padding=k // 2, bias=False),
                nl(start_filter), nn.ReLU(inplace=True))

        if stem_type == "v0":
            self.stem = cbr(in_chans, 7)
        elif stem_type == "v1":
            self.stem = nn.Sequential(cbr(in_chans, 3), cbr(start_filter, 3), cbr(start_filter, 3))
        else:
            raise ValueError(f"stem_type must be 'v0' or 'v1', got {stem_type!r}")
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        num_lvls = len(block_repeats)
        self.groups = nn.ModuleList()
        self.output_specs = {}
        if expose_stem:
            self.output_specs["1"] = start_filter
        in_ch = start_filter
        for i, (f, r) in enumerate(zip(num_filters, block_repeats)):
            stride = 1 if i == 0 else 2
            dp = stochastic_depth_rate * (i + 2) / (num_lvls + 1) if stochastic_depth_rate else 0.0
            blocks = [BasicBlock(in_ch, f, stride, use_projection=True, drop_path_rate=dp,
                                 norm_layer=norm_layer)]
            blocks += [BasicBlock(f, f, 1, use_projection=False, drop_path_rate=dp,
                                  norm_layer=norm_layer)
                       for _ in range(1, r)]
            self.groups.append(nn.Sequential(*blocks))
            in_ch = f
            self.output_specs[str(i + 2)] = f

    def forward_features(self, x):
        x = self.stem(x)
        out = {"1": x} if self.expose_stem else {}
        x = self.maxpool(x)
        for i, group in enumerate(self.groups):
            x = group(x)
            out[str(i + 2)] = x
        return out

    forward = forward_features


# --------------------------------------------------------------------------- DINOv3 ConvNeXt (timm)
class ConvNeXtDinoV3Features(nn.Module):
    """DINOv3-pretrained ConvNeXt-Tiny as a 4-stage feature pyramid (timm ``features_only``).

    Exposes the same ``forward_features``/``output_specs`` contract as the other encoders: stages map
    to endpoints ``{"2","3","4","5"}`` at strides 4/8/16/32 (level = log2 stride), channels derived
    from ``feature_info`` (not hardcoded, so a different timm variant just works).

    2-channel stem: we do NOT use timm's default ``in_chans=2`` adapt (which repeat-slices R/G).
    Instead we load the pretrained 3-ch model, then replace the patch-embed stem conv with
    ``(3/in_chans) * mean_over_RGB`` broadcast across the input channels -- a luminance-style filter
    (sensible for grayscale-ish fluorescence), with the ``3/in_chans`` factor preserving the stem's
    output magnitude that the frozen downstream trunk was calibrated for.
    """

    def __init__(self, in_chans: int = 2, pretrained: bool = True,
                 model_name: str = "convnext_tiny.dinov3_lvd1689m"):
        super().__init__()
        import math
        import timm  # lazy: timm is an optional dep, only needed for this encoder
        # build at in_chans=3 so pretrained weights load cleanly, then convert the stem
        self.body = timm.create_model(model_name, pretrained=pretrained, features_only=True,
                                      out_indices=(0, 1, 2, 3), in_chans=3)
        if in_chans != 3:
            self._convert_stem(in_chans)
        chans = list(self.body.feature_info.channels())     # [96,192,384,768]
        reds = list(self.body.feature_info.reduction())      # [4,8,16,32]
        self._levels = [str(int(round(math.log2(r)))) for r in reds]   # ["2","3","4","5"]
        self.output_specs = {lvl: c for lvl, c in zip(self._levels, chans)}

    def _find_stem_conv(self):
        """The patch-embed stem is the unique Conv2d that takes the raw 3-ch input."""
        for name, m in self.body.named_modules():
            if isinstance(m, nn.Conv2d) and m.in_channels == 3:
                return name, m
        raise RuntimeError("could not locate ConvNeXt stem conv (in_channels==3)")

    def _set_module(self, dotted: str, new: nn.Module):
        parent = self.body
        parts = dotted.split(".")
        for p in parts[:-1]:
            parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
        last = parts[-1]
        if last.isdigit():
            parent[int(last)] = new
        else:
            setattr(parent, last, new)

    def _convert_stem(self, in_chans: int):
        name, conv = self._find_stem_conv()
        w = conv.weight.data                                  # [out, 3, k, k]
        mean_w = w.mean(dim=1, keepdim=True)                  # [out, 1, k, k]
        new_w = (3.0 / in_chans) * mean_w.repeat(1, in_chans, 1, 1)   # scaled-mean, both channels equal
        new_conv = nn.Conv2d(in_chans, conv.out_channels, conv.kernel_size, stride=conv.stride,
                             padding=conv.padding, dilation=conv.dilation, groups=conv.groups,
                             bias=(conv.bias is not None))
        new_conv.weight.data.copy_(new_w)
        if conv.bias is not None:
            new_conv.bias.data.copy_(conv.bias.data)
        self._set_module(name, new_conv)

    def forward_features(self, x):
        feats = self.body(x)                                  # list of 4 stage maps
        return {lvl: f for lvl, f in zip(self._levels, feats)}

    forward = forward_features


# --------------------------------------------------------------------------- parallel high-res branch
class HiResBranch(nn.Module):
    """Shallow parallel high-resolution branch from the raw input -> (f0 @ stride-1, f1 @ stride-2).

    Uses GroupNorm (batch-independent, no BN running-stats / eval-mode artifacts). Deliberately
    shallow/narrow with a small receptive field so it CANNOT substitute for the encoder's global
    context -- this keeps the DINOv3-contribution ablation (frozen-with-ConvNeXt vs hi-res-only)
    meaningful.
    """

    def __init__(self, in_chans: int = 2, c0: int = 32, c1: int = 32, gn_groups: int = 8):
        super().__init__()
        gn = make_gn(gn_groups)

        def cbr(cin, cout, stride):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False),
                gn(cout), nn.ReLU(inplace=True))

        self.stem0 = nn.Sequential(cbr(in_chans, c0, 1), cbr(c0, c0, 1))   # stride-1, full res
        self.down = cbr(c0, c1, 2)                                         # stride-2, half res

    def forward(self, x):
        f0 = self.stem0(x)      # [N, c0, H, W]
        f1 = self.down(f0)      # [N, c1, H/2, W/2]
        return f0, f1


# --------------------------------------------------------------------- ConvNeXt-V2 (FCMAE) 5-stage
# Standalone DENSE ConvNeXt-V2 trunk (no MinkowskiEngine, no timm) whose module layout is key-identical
# to the official dense ConvNeXtV2, so an FCMAE-pretrained sparse encoder converted via the repo's
# sparse->dense remap_checkpoint_keys loads 1:1 (downsample_layers.*/stages.*). Exposes the standard
# forward_features/output_specs contract at native levels "1".."5" (strides 2/4/8/16/32) -- the stride-2
# stem gives a native level-1 endpoint, so no parallel hi-res branch is needed (RidgepathSegModel with
# fpn_min_level=1, head_level=1, final bilinear upsample handles full-res, like the tenxnet_recipe path).

class _CNX2LayerNorm(nn.Module):
    """LayerNorm supporting channels_last (N,H,W,C) or channels_first (N,C,H,W) -- matches official."""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class _CNX2GRN(nn.Module):
    """Global Response Normalization (ConvNeXt-V2), channels_last; gamma/beta shape [1,1,1,dim]."""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class _CNX2Block(nn.Module):
    """ConvNeXt-V2 block: 7x7 dw -> LN -> Linear C->4C -> GELU -> GRN -> Linear 4C->C -> residual."""
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = _CNX2LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = _CNX2GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)          # NCHW -> NHWC
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)          # NHWC -> NCHW
        return shortcut + x


class ConvNeXtV2Features(nn.Module):
    """Dense N-stage ConvNeXt-V2 feature-pyramid encoder (defaults = the FCMAE atto 5-stage config).

    forward_features -> {"1".."N"} at strides stem_stride, 2*stem_stride, ... ; output_specs = {level: C}.
    No final norm/head (matches the FCMAE encoder subtree exactly), so a converted dense state_dict of
    keys downsample_layers.*/stages.* loads with 0 missing.
    """
    def __init__(self, in_chans: int = 2, dims=(20, 40, 80, 160, 320), depths=(2, 2, 2, 6, 2),
                 stem_stride: int = 2):
        super().__init__()
        dims, depths = list(dims), list(depths)
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Sequential(          # non-overlapping stride-2 stem
            nn.Conv2d(in_chans, dims[0], kernel_size=stem_stride, stride=stem_stride),
            _CNX2LayerNorm(dims[0], eps=1e-6, data_format="channels_first")))
        for i in range(len(dims) - 1):
            self.downsample_layers.append(nn.Sequential(
                _CNX2LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)))
        self.stages = nn.ModuleList(
            nn.Sequential(*[_CNX2Block(dims[i]) for _ in range(depths[i])]) for i in range(len(depths)))
        self._levels = [str(i + 1) for i in range(len(dims))]     # stem_stride=2 -> "1".."5" = s2..s32
        self.output_specs = {lvl: c for lvl, c in zip(self._levels, dims)}

    def forward_features(self, x):
        out = {}
        for i, lvl in enumerate(self._levels):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            out[lvl] = x
        return out

    forward = forward_features


def build_encoder(name: str, in_chans: int, encoder_norm: str = "bn", gn_max_groups: int = 32,
                  **enc_kwargs):
    """Build a feature-pyramid encoder.

    ``encoder_norm='bn'`` (default) keeps the original BatchNorm path exactly. ``'gn'`` swaps every
    encoder norm to adaptive GroupNorm (via :func:`make_gn`) -- used for the iBOT/DINOv2 SSL encoder
    so it (a) has no batch-coupled running stats and (b) matches the GN SSL backbone key-for-key.
    Only the tenxnet encoders support GN (resnet18 stays BN; it is not used for the SSL->seg handoff).
    """
    if encoder_norm not in ("bn", "gn"):
        raise ValueError(f"encoder_norm must be 'bn' or 'gn', got {encoder_norm!r}")
    nl = make_gn(gn_max_groups) if encoder_norm == "gn" else None
    if name == "resnet18":
        if encoder_norm == "gn":
            raise ValueError("encoder_norm='gn' is not supported for resnet18 (BN-only drop-in)")
        return ResNet18Features(in_chans)
    if name == "tenxnet_small":  # v0 stem, no stochastic depth, endpoints 2..5 (FPN min_level 2)
        return TenxnetSmallResNet(in_chans, norm_layer=nl)
    if name == "tenxnet_recipe":  # faithful test.yaml: v1 stem, SD 0.5, endpoints 1..5 (FPN min_level 1)
        return TenxnetSmallResNet(in_chans, stem_type="v1", stochastic_depth_rate=0.5,
                                  expose_stem=True, norm_layer=nl)
    if name == "convnext_dino":   # DINOv3 ConvNeXt-Tiny (timm), endpoints 2..5 @ stride 4/8/16/32
        return ConvNeXtDinoV3Features(
            in_chans, pretrained=enc_kwargs.get("pretrained", True),
            model_name=enc_kwargs.get("model_name", "convnext_tiny.dinov3_lvd1689m"))
    if name == "convnextv2_5stage":   # FCMAE-pretrained dense ConvNeXt-V2, endpoints 1..5 @ stride 2/4/8/16/32
        return ConvNeXtV2Features(
            in_chans, dims=enc_kwargs.get("dims", (20, 40, 80, 160, 320)),
            depths=enc_kwargs.get("depths", (2, 2, 2, 6, 2)), stem_stride=enc_kwargs.get("stem_stride", 2))
    raise ValueError(f"unknown encoder {name!r} (expected 'resnet18', 'tenxnet_small', "
                     f"'tenxnet_recipe', 'convnext_dino', or 'convnextv2_5stage')")
