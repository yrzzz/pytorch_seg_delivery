"""PyTorch port of tenxnet EncoderDecoderHead (tenxnet/vision/models/encoder_decoder_head.py).

Assembles encoder -> FPN -> SegmentationHead and emits the 9-channel Ridgepath logits at
**input resolution** (the targets are full-resolution).

Resolution note: tenxnet's small encoder has a stride-1 stem, so head fusion to ``level=1``
yields full-res output natively (final resize is a no-op). The drop-in ResNet18 has a stride-2
stem (no level-1 feature), so we fuse to the lowest FPN level and the assembly does ONE final
``F.interpolate`` to the input H x W. ``build_seg_model`` picks the right head level + final
mode per backbone.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import HiResBranch, build_encoder
from .fpn import FPN
from .layers import make_gn
from .seg_head import SegmentationHead, pyramid_feature_fusion


class RidgepathSegModel(nn.Module):
    def __init__(self, encoder_name="resnet18", in_chans=2, num_classes=9,
                 fpn_min_level=2, fpn_max_level=5, fpn_filters=32,
                 head_level=2, head_filters=32, head_num_convs=2,
                 prediction_kernel_size=1, final_upsample_mode="bilinear",
                 encoder_norm="bn", gn_max_groups=32, enc_kwargs=None):
        super().__init__()
        # Only the ENCODER may switch to GroupNorm (SSL->seg handoff needs matching norm/keys); the
        # FPN + head stay BatchNorm (never masked, never cross the SSL boundary). enc_kwargs forwards
        # encoder-specific options (e.g. convnext_dino: pretrained/model_name) to build_encoder.
        self.encoder = build_encoder(encoder_name, in_chans, encoder_norm=encoder_norm,
                                     gn_max_groups=gn_max_groups, **(enc_kwargs or {}))
        self.fpn = FPN(self.encoder.output_specs, fpn_min_level, fpn_max_level, fpn_filters)
        self.head = SegmentationHead(
            num_classes=num_classes, level=head_level, in_channels=fpn_filters,
            num_convs=head_num_convs, num_filters=head_filters,
            prediction_kernel_size=prediction_kernel_size, feature_fusion="pyramid_fusion",
        )
        self.final_upsample_mode = final_upsample_mode

    def forward(self, x):
        in_hw = x.shape[-2:]
        feats = self.encoder.forward_features(x)
        dec = self.fpn(feats)
        out = self.head(dec)
        if out.shape[-2:] != in_hw:
            kw = {} if self.final_upsample_mode == "nearest" else {"align_corners": False}
            out = F.interpolate(out, size=in_hw, mode=self.final_upsample_mode, **kw)
        return out


class ConvNeXtHiResSegModel(nn.Module):
    """DINOv3 ConvNeXt-Tiny (frozen) + parallel high-res branch -> 5-level FPN -> late stride-1
    fusion -> Ridgepath head. See plan: keep ``self.encoder`` = ConvNeXt ONLY so the existing
    freeze/optimizer machinery in train_seg.py treats it atomically; hi-res/fpn/fuse/head are
    siblings (decoder param group).

    Levels (level = log2 stride): hi-res stride-2 = level 1; ConvNeXt strides 4/8/16/32 = levels
    2..5. The stride-1 hi-res feature is NOT in the FPN -- it is fused in late, after upsampling the
    FPN's finest (level-1 / stride-2) output to full res, so the head emits a full-res 9-ch map.

    ``use_convnext=False`` -> hi-res-only ablation: ``self.encoder=nn.Identity()``, FPN sees only
    level 1, everything else identical.
    """

    def __init__(self, in_chans=2, num_classes=9, use_convnext=True, pretrained=True,
                 timm_model="convnext_tiny.dinov3_lvd1689m", fpn_filters=32,
                 head_filters=32, head_num_convs=2, prediction_kernel_size=1,
                 hires_c0=32, hires_c1=32, gn_groups=8, **_):
        super().__init__()
        self.use_convnext = use_convnext
        self.hires = HiResBranch(in_chans, c0=hires_c0, c1=hires_c1, gn_groups=gn_groups)
        if use_convnext:
            self.encoder = build_encoder("convnext_dino", in_chans, pretrained=pretrained,
                                         model_name=timm_model)
            fpn_specs = {"1": hires_c1, **self.encoder.output_specs}
        else:
            self.encoder = nn.Identity()      # ablation: no ConvNeXt; empty encoder param group
            fpn_specs = {"1": hires_c1}
        levels = sorted(int(k) for k in fpn_specs)
        self.fpn_min_level = levels[0]
        self.fpn = FPN(fpn_specs, min_level=levels[0], max_level=levels[-1], num_filters=fpn_filters)
        self.hires_proj = nn.Conv2d(hires_c0, fpn_filters, kernel_size=1)   # stride-1 -> fpn ch (late)
        gn = make_gn(gn_groups)
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * fpn_filters, fpn_filters, kernel_size=3, padding=1, bias=False),
            gn(fpn_filters), nn.ReLU(inplace=True))
        self.head = SegmentationHead(
            num_classes=num_classes, level=0, in_channels=fpn_filters, num_convs=head_num_convs,
            num_filters=head_filters, prediction_kernel_size=prediction_kernel_size,
            feature_fusion="single_level")   # any non-"pyramid_fusion" -> consume decoder["0"] verbatim

    def forward(self, x):
        f0, f1 = self.hires(x)                                  # stride-1, stride-2
        feats = {"1": f1}
        if self.use_convnext:
            feats.update(self.encoder.forward_features(x))      # {"2".."5"}
        dec = self.fpn(feats)
        p1 = pyramid_feature_fusion(dec, self.fpn_min_level)    # sum FPN levels -> stride-2 (level 1)
        up = F.interpolate(p1, size=f0.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.fuse(torch.cat([up, self.hires_proj(f0)], dim=1))   # full-res fusion
        return self.head({"0": fused})                          # (N, num_classes, H, W)


def build_seg_model(encoder_name="resnet18", in_chans=2, num_classes=9,
                    encoder_norm="bn", gn_max_groups=32, **overrides):
    """Construct a RidgepathSegModel with backbone-appropriate resolution defaults.

    ``encoder_norm='gn'`` builds a GroupNorm encoder (for loading an iBOT/DINOv2 GN-pretrained
    checkpoint); default ``'bn'`` is unchanged.

    ``encoder_name in {"convnext_dino_hires","hires_only"}`` builds the ConvNeXt+hi-res model
    instead (``overrides`` carries model_kwargs: pretrained/timm_model/hires_c0/hires_c1/...).
    """
    if encoder_name in ("convnext_dino_hires", "hires_only"):
        return ConvNeXtHiResSegModel(in_chans=in_chans, num_classes=num_classes,
                                     use_convnext=(encoder_name == "convnext_dino_hires"),
                                     **overrides)
    if encoder_name == "convnext_dino":
        # DINOv3 ConvNeXt-Tiny 4-stage backbone ALONE (no hi-res branch): 4-stage FPN over P2-P5
        # (strides 4/8/16/32) -> head at level 2 (stride 4) -> one bilinear x4 to full res. model_kwargs
        # (pretrained/model_name) forward to the encoder; `timm_model` is accepted as a model_name alias.
        mk = dict(overrides)
        if "timm_model" in mk:
            mk["model_name"] = mk.pop("timm_model")
        return RidgepathSegModel(
            encoder_name="convnext_dino", in_chans=in_chans, num_classes=num_classes,
            fpn_min_level=2, fpn_max_level=5, fpn_filters=32, head_level=2, head_filters=32,
            head_num_convs=2, prediction_kernel_size=1, final_upsample_mode="bilinear",
            encoder_norm=encoder_norm, gn_max_groups=gn_max_groups, enc_kwargs=mk)
    defaults = dict(fpn_min_level=2, fpn_max_level=5, fpn_filters=32,
                    head_filters=32, head_num_convs=2, prediction_kernel_size=1,
                    encoder_norm=encoder_norm, gn_max_groups=gn_max_groups)
    if encoder_name in ("tenxnet_recipe", "convnextv2_5stage"):
        # native level-1 (stride-2) endpoint: FPN spans P1-P5, fuse to level 1, one bilinear x2 -> full res.
        defaults.update(fpn_min_level=1, head_level=1, final_upsample_mode="bilinear")
    elif encoder_name == "tenxnet_small":
        # stride-1 stem: level "2" is stride 2, fuse to level 1 -> full res.
        defaults.update(head_level=1, final_upsample_mode="bilinear")
    else:  # resnet18 drop-in: lowest FPN level is /4, resize to input.
        defaults.update(head_level=2, final_upsample_mode="bilinear")
    defaults.update(overrides)
    return RidgepathSegModel(encoder_name=encoder_name, in_chans=in_chans,
                             num_classes=num_classes, **defaults)
