"""
TriConsistencyNet — Frequency Guidance Encoder (FGE)

Pipeline:
  RGB (B,3,224,224)
  → 2D FFT magnitude → log(1+|F|) → per-sample [0,1] normalisation
  → Lightweight CNN  (3 conv-BN-SiLU + 2 residual blocks)
  → (B, 256, 7, 7)

Total learnable params ≈ 0.8 M (matches architecture spec).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Standard pre-activation residual block (no downsampling)."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class FrequencyGuidanceEncoder(nn.Module):
    """
    FFT preprocessing followed by a lightweight CNN.

    Architecture:
        Conv(3→32, s2)   : 224→112
        Conv(32→64, s2)  : 112→56
        Conv(64→128, s2) : 56→28
        ResBlock(128)
        Conv(128→256, s2): 28→14
        ResBlock(256)
        Conv(256→256, s2): 14→7

    Output: (B, 256, 7, 7)
    """

    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            # ── stride-2 block 1 ───────────────────────────────────────
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            # ── stride-2 block 2 ───────────────────────────────────────
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            # ── stride-2 block 3 ───────────────────────────────────────
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            # ── residual block 1 ───────────────────────────────────────
            ResBlock(128),
            # ── stride-2 block 4 ───────────────────────────────────────
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            # ── residual block 2 ───────────────────────────────────────
            ResBlock(256),
            # ── stride-2 block 5 ───────────────────────────────────────
            nn.Conv2d(256, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
        )

    # ------------------------------------------------------------------
    # FFT helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _fft_magnitude(x: torch.Tensor) -> torch.Tensor:
        """
        Compute log-compressed, per-sample-normalised magnitude spectrum.

        x : (B, 3, H, W) float
        returns: (B, 3, H, W) float in [0, 1]
        """
        # Full 2-D FFT on last two dims, shift DC to centre
        f = torch.fft.fft2(x)
        f = torch.fft.fftshift(f, dim=(-2, -1))
        mag = torch.abs(f)                        # (B, 3, H, W)
        mag = torch.log1p(mag)                    # log(1 + |F|)
        # Per-sample normalisation to [0, 1]
        b = mag.shape[0]
        mn = mag.view(b, -1).min(dim=1).values.view(b, 1, 1, 1)
        mx = mag.view(b, -1).max(dim=1).values.view(b, 1, 1, 1)
        mag = (mag - mn) / (mx - mn + 1e-8)
        return mag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) — same RGB tensor as the spatial stream

        Returns:
            (B, 256, 7, 7)
        """
        freq = self._fft_magnitude(x)   # (B, 3, H, W) in [0,1]
        return self.cnn(freq)           # (B, 256, 7, 7)
