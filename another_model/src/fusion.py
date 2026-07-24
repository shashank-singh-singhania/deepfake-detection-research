"""
TriConsistencyNet — Adaptive Feature Fusion (AFF)

Squeeze-and-Excitation style channel calibration:

  S_refined (B, 1280, 7, 7)
    → Conv1×1 + BN + SiLU            (cross-channel mixing)
    → Global Average Pool             (B, 1280)
    → SE bottleneck: 1280 → 320 → 1280 + Sigmoid gate
    → v_fused = pooled ⊙ gate        (B, 1280)

Returns:
    (B, 1280) — ready for Dropout + Linear classifier
"""

import torch
import torch.nn as nn


class AdaptiveFeatureFusion(nn.Module):

    def __init__(self, channels: int = 1280, reduction: int = 4):
        super().__init__()
        mid = channels // reduction   # 320

        # Channel mixing before pooling
        self.mix = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

        # SE bottleneck gate (operates on the pooled 1-D vector)
        self.se = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1280, 7, 7)

        Returns:
            (B, 1280)
        """
        x = self.mix(x)                       # (B, 1280, 7, 7)
        pooled = x.mean(dim=(-2, -1))          # (B, 1280) — GAP
        gate = self.se(pooled)                 # (B, 1280) ∈ [0, 1]
        return pooled * gate                   # (B, 1280)
