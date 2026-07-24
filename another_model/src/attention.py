"""
TriConsistencyNet — Cross-Consistency Attention (CCA)

Both spatial S:(B,1280,7,7) and frequency F:(B,256,7,7) are projected
into a shared 1280-d channel space. Element-wise multiplication yields
a "consistency map" — activated only where both streams agree. A gating
network converts this into a spatial attention map A∈[0,1].

Residual formulation: S_refined = S ⊙ (1 + A)
  → If A=0 everywhere, the spatial features pass unchanged (no collapse).

Returns:
    refined  : (B, 1280, 7, 7)
    attention: (B,    1, 7, 7)  — for visualisation / diagnostics
"""

import torch
import torch.nn as nn


class CrossConsistencyAttention(nn.Module):

    def __init__(self, spatial_ch: int = 1280, freq_ch: int = 256, hidden: int = 1280):
        super().__init__()

        # Project spatial stream → hidden_ch
        self.proj_s = nn.Sequential(
            nn.Conv2d(spatial_ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )

        # Project frequency stream → hidden_ch
        self.proj_f = nn.Sequential(
            nn.Conv2d(freq_ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )

        # Gating network: consistency map → scalar attention map
        # Two 1×1 convs: hidden → hidden//8 → 1
        gate_mid = max(hidden // 8, 16)
        self.gate = nn.Sequential(
            nn.Conv2d(hidden, gate_mid, 1, bias=False),
            nn.BatchNorm2d(gate_mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(gate_mid, 1, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(
        self,
        spatial: torch.Tensor,   # (B, 1280, 7, 7)
        frequency: torch.Tensor, # (B,  256, 7, 7)
    ):
        s_proj = self.proj_s(spatial)      # (B, 1280, 7, 7)
        f_proj = self.proj_f(frequency)    # (B, 1280, 7, 7)

        consistency = s_proj * f_proj      # element-wise product

        attention = self.gate(consistency) # (B, 1, 7, 7)  ∈ [0, 1]

        refined = spatial * (1.0 + attention)  # residual refinement

        return refined, attention
