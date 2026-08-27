"""Shared layer helpers for the ported seg model.

TF BatchNormalization uses momentum=0.99 (moving-average decay) and epsilon=1e-3. PyTorch's
BatchNorm momentum has the opposite convention (running = (1-m)*old + m*new), so the equivalent
PyTorch momentum is ``1 - 0.99 = 0.01``.
"""
import torch.nn as nn

BN_MOMENTUM = 0.01  # == TF norm_momentum 0.99
BN_EPS = 1e-3       # == TF norm_epsilon 0.001

GN_EPS = 1e-5       # nn.GroupNorm default; GN has no running stats so the TF-parity eps is irrelevant
GN_MAX_GROUPS = 32  # DINOv2/iBOT default group cap


def bn(num_features: int) -> nn.BatchNorm2d:
    return nn.BatchNorm2d(num_features, momentum=BN_MOMENTUM, eps=BN_EPS)


def adaptive_gn_groups(num_channels: int, max_groups: int = GN_MAX_GROUPS) -> int:
    """Largest divisor of ``num_channels`` that is ``<= max_groups`` (guaranteed to divide evenly).

    Used to pick a GroupNorm group count that (a) divides every stage width and (b) stays near the
    DINOv2 default of 32. For the tenxnet widths this yields 16->16, 32->32, 64->32, 128->32.
    """
    if num_channels <= 0:
        raise ValueError(f"num_channels must be positive, got {num_channels}")
    for g in range(min(max_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1  # unreachable (g==1 always divides)


def make_gn(max_groups: int = GN_MAX_GROUPS, eps: float = GN_EPS, norm_cls=nn.GroupNorm):
    """Return a ``num_features -> GroupNorm`` factory with the same signature as ``bn``.

    The group count is chosen by :func:`adaptive_gn_groups` and MUST match on both sides of the
    SSL->segmentation handoff, so callers pass the *same* ``max_groups`` for the SSL backbone and the
    seg encoder. ``norm_cls`` lets the SSL side substitute a mask-aware GroupNorm subclass while
    keeping identical parameters/keys. The chosen ``(channels -> groups)`` map is printed once per
    unique width so the architecture is auditable.
    """
    seen = {}

    def factory(num_features: int):
        groups = adaptive_gn_groups(num_features, max_groups)
        if num_features not in seen:
            seen[num_features] = groups
            print(f"[make_gn] GroupNorm(groups={groups}, channels={num_features}) "
                  f"[{num_features // groups} ch/group]")
        return norm_cls(groups, num_features, eps=eps)

    return factory
