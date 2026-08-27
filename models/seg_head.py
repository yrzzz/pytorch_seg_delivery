"""PyTorch port of tenxnet SegmentationHead (tenxnet/vision/models/heads/segmentation_heads.py).

  * ``pyramid_feature_fusion``: bilinear-resample every pyramid level to ``target_level`` and sum.
  * ``num_convs`` x (conv3x3 [no bias] -> BN -> ReLU),
  * optional nearest upsample by ``upsample_factor``,
  * final 1x1 classifier conv (bias, zero-init) -> ``num_classes`` logits (no activation).

Init matches TF: head convs / classifier kernels use Normal(std=0.01); classifier bias = 0.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import bn


def pyramid_feature_fusion(inputs: dict, target_level: int) -> torch.Tensor:
    """Resample all pyramid levels to ``target_level`` (bilinear) and sum (== TF add_n)."""
    levels = sorted(int(k) for k in inputs)
    min_level, max_level = levels[0], levels[-1]
    target_hw = inputs[str(target_level)].shape[-2:]
    resampled = []
    for lvl in range(min_level, max_level + 1):
        feat = inputs[str(lvl)]
        if lvl == target_level:
            resampled.append(feat)
        else:
            resampled.append(
                F.interpolate(feat, size=target_hw, mode="bilinear", align_corners=False)
            )
    return torch.stack(resampled, dim=0).sum(dim=0)


class SegmentationHead(nn.Module):
    def __init__(self, num_classes: int, level: int, in_channels: int, num_convs: int = 2,
                 num_filters: int = 256, prediction_kernel_size: int = 1,
                 upsample_factor: int = 1, feature_fusion: str = "pyramid_fusion"):
        super().__init__()
        self.num_classes = num_classes
        self.level = level
        self.num_filters = num_filters
        self.upsample_factor = upsample_factor
        self.feature_fusion = feature_fusion

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        ch = in_channels
        for _ in range(num_convs):
            self.convs.append(nn.Conv2d(ch, num_filters, kernel_size=3, padding=1, bias=False))
            self.norms.append(bn(num_filters))
            ch = num_filters
        self.classifier = nn.Conv2d(
            num_filters, num_classes, kernel_size=prediction_kernel_size,
            padding=prediction_kernel_size // 2, bias=True,
        )
        self._init_weights()

    def _init_weights(self):
        for m in list(self.convs) + [self.classifier]:
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, decoder_output: dict) -> torch.Tensor:
        if self.feature_fusion == "pyramid_fusion":
            x = pyramid_feature_fusion(decoder_output, self.level)
        else:
            x = decoder_output[str(self.level)]
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(conv(x)))
        if self.upsample_factor > 1:
            x = F.interpolate(x, scale_factor=self.upsample_factor, mode="nearest")
        return self.classifier(x)
