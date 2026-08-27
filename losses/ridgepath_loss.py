"""PyTorch port of tenxnet ``RidgepathLoss`` (tenxnet/vision/models/losses/ridgepath_loss.py).

The TF original (verbatim semantics):

    segment_loss = tf.nn.sigmoid_cross_entropy_with_logits(targets[...,0], inputs[...,0])
    row_dir_loss = tf.nn.softmax_cross_entropy_with_logits(labels=targets[...,1:5], logits=inputs[...,1:5])
    col_dir_loss = tf.nn.softmax_cross_entropy_with_logits(labels=targets[...,5:9], logits=inputs[...,5:9])
    loss = reduce_mean(segment_loss)
         + reduce_mean(row_dir_loss * targets[...,9])
         + reduce_mean(col_dir_loss * targets[...,9])

Two port-critical details:
  * ``softmax_cross_entropy_with_logits`` takes **soft label distributions**, not class
    indices -> use ``-(target * log_softmax(logit)).sum(channel)``; NOT ``F.cross_entropy``.
  * The semantic term is **unweighted**; only the row/col direction terms are multiplied by
    the per-pixel weight (channel 9). All three terms reduce by ``mean`` over B*H*W.

This module operates on **NCHW** tensors (PyTorch-native): logits ``[B, 9, H, W]`` and
targets ``[B, 10, H, W]`` with the channel layout
``[semantic, row0..row3, col0..col3, weight]``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def ridgepath_loss(logits: torch.Tensor, targets: torch.Tensor):
    """Compute the ridgepath loss. Returns (total, segment, row, col) scalar tensors.

    Args:
        logits: model output, ``[B, 9, H, W]`` (raw logits, channel dim = 1).
        targets: ground-truth ridgepath label, ``[B, 10, H, W]`` (channel dim = 1).
    """
    if logits.shape[1] != 9:
        raise ValueError(f"expected 9 logit channels (dim=1), got shape {tuple(logits.shape)}")
    if targets.shape[1] != 10:
        raise ValueError(f"expected 10 target channels (dim=1), got shape {tuple(targets.shape)}")

    # Numerical guard for AMP/fp16: cap logits so the forward can't overflow fp16 (max ~65504) to inf->NaN.
    # softmax/sigmoid already saturate by ~±15, so clamping at ±30 leaves predictions/gradients unchanged
    # for any reasonable logit but keeps the loss finite even if a logit blows up. (fp16 NaN was a forward
    # overflow that grad-clipping cannot catch; this is the actual fix.)
    logits = logits.clamp(-30.0, 30.0)

    # Semantic: sigmoid cross-entropy, unweighted, mean over all pixels.
    segment_loss = F.binary_cross_entropy_with_logits(
        logits[:, 0], targets[:, 0], reduction="mean"
    )

    weight = targets[:, 9]  # [B, H, W]

    # Row directions (channels 1:5): soft-target CE over the 4-channel group.
    row_ce = -(targets[:, 1:5] * F.log_softmax(logits[:, 1:5], dim=1)).sum(dim=1)  # [B,H,W]
    row_dir_loss = (row_ce * weight).mean()

    # Col directions (channels 5:9): same.
    col_ce = -(targets[:, 5:9] * F.log_softmax(logits[:, 5:9], dim=1)).sum(dim=1)  # [B,H,W]
    col_dir_loss = (col_ce * weight).mean()

    total = segment_loss + row_dir_loss + col_dir_loss
    return total, segment_loss, row_dir_loss, col_dir_loss


class RidgepathLoss(nn.Module):
    """nn.Module wrapper mirroring tenxnet's ``RidgepathLoss`` class."""

    name = "Ridgepath_Loss"

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total, _, _, _ = ridgepath_loss(logits, targets)
        return total
