"""PyTorch port of tenxnet FPN (tenxnet/vision/models/decoders/fpn.py).

Mirrors the TF FPN exactly:
  * lateral 1x1 conv per backbone level (min_level..backbone_max_level),
  * top-down nearest-upsampling (x2) + add,
  * post-hoc 3x3 smoothing conv per level,
  * BatchNorm on every output level.

Built from the encoder's ``output_specs`` ({level: channels}), exactly like the TF FPN is built
from ``input_specs``. ``coarse_level`` (FPN levels beyond the backbone) is unused in the ridgepath
config (max_level == backbone max) and is not supported here.
"""
import torch.nn as nn
import torch.nn.functional as F

from .layers import bn


class FPN(nn.Module):
    def __init__(self, input_specs: dict, min_level: int = 2, max_level: int = 5,
                 num_filters: int = 256):
        super().__init__()
        levels = sorted(int(k) for k in input_specs)
        self.min_level = min_level
        self.backbone_max_level = min(max(levels), max_level)
        self.max_level = max_level
        if self.max_level != self.backbone_max_level:
            raise ValueError("coarse FPN levels (max_level > backbone max) not supported")
        self.num_filters = num_filters

        self.lateral = nn.ModuleDict()
        self.smooth = nn.ModuleDict()
        self.norms = nn.ModuleDict()
        for lvl in range(min_level, self.backbone_max_level + 1):
            in_ch = input_specs[str(lvl)]
            # TF Conv2D defaults to use_bias=True for lateral + smoothing convs.
            self.lateral[str(lvl)] = nn.Conv2d(in_ch, num_filters, kernel_size=1, bias=True)
            self.smooth[str(lvl)] = nn.Conv2d(num_filters, num_filters, kernel_size=3,
                                              padding=1, bias=True)
        for lvl in range(min_level, self.max_level + 1):
            self.norms[str(lvl)] = bn(num_filters)

    def forward(self, feats: dict) -> dict:
        lat = {l: self.lateral[l](feats[l]) for l in self.lateral}
        out = {str(self.backbone_max_level): lat[str(self.backbone_max_level)]}
        for lvl in range(self.backbone_max_level - 1, self.min_level - 1, -1):
            # nearest x2; use target size for robustness to odd dims (== x2 for even dims).
            up = F.interpolate(out[str(lvl + 1)], size=lat[str(lvl)].shape[-2:], mode="nearest")
            out[str(lvl)] = up + lat[str(lvl)]
        for lvl in range(self.min_level, self.backbone_max_level + 1):
            out[str(lvl)] = self.smooth[str(lvl)](out[str(lvl)])
        for lvl in range(self.min_level, self.max_level + 1):
            out[str(lvl)] = self.norms[str(lvl)](out[str(lvl)])
        return out
