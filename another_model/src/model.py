"""
TriConsistencyNet — Complete Model (self-contained, no config files)

Architecture:
  EfficientNetV2-S (frozen backbone) → spatial (B, 1280, 7, 7)
  FrequencyGuidanceEncoder           → frequency (B,  256, 7, 7)
  CrossConsistencyAttention          → refined (B, 1280, 7, 7) + attention map
  AdaptiveFeatureFusion              → features (B, 1280)
  Dropout(0.3) + Linear(1280→1)      → logit  (B, 1)  [BCEWithLogitsLoss]

Output is a single logit. Use sigmoid for probability.
The attention map is stored in self.last_attention_map for visualisation.

Parameter count breakdown:
  EfficientNetV2-S (frozen):   ~20.3 M  (0 trainable)
  FGE CNN:                      ~0.8 M  (trainable)
  CCA:                          ~3.3 M  (trainable)
  AFF:                          ~3.3 M  (trainable)
  Classifier head:              ~1.3 K  (trainable)
  ─────────────────────────────────────────────────
  Total:                       ~27.7 M  | Trainable: ~7.4 M (≈27 %)
"""

from pathlib import Path
import sys

# Ensure the another_model directory is resolvable as a standalone package
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import torch
import torch.nn as nn
import timm

from frequency import FrequencyGuidanceEncoder
from attention import CrossConsistencyAttention
from fusion    import AdaptiveFeatureFusion


class TriConsistencyNet(nn.Module):

    SPATIAL_CH = 1280   # EfficientNetV2-S penultimate feature channels
    FREQ_CH    = 256

    def __init__(self, freeze_backbone: bool = True, dropout: float = 0.3):
        super().__init__()

        # ── Spatial stream ────────────────────────────────────────────
        # global_pool="" keeps the spatial (H,W) dims intact
        self.backbone = timm.create_model(
            "tf_efficientnetv2_s",
            pretrained=True,
            num_classes=0,
            global_pool="",
        )
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)

        # ── Frequency stream ──────────────────────────────────────────
        self.fge = FrequencyGuidanceEncoder()

        # ── Cross-Consistency Attention ───────────────────────────────
        self.cca = CrossConsistencyAttention(
            spatial_ch=self.SPATIAL_CH,
            freq_ch=self.FREQ_CH,
            hidden=self.SPATIAL_CH,
        )

        # ── Adaptive Feature Fusion ───────────────────────────────────
        self.aff = AdaptiveFeatureFusion(channels=self.SPATIAL_CH, reduction=4)

        # ── Classifier ───────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.SPATIAL_CH, 1),  # BCEWithLogitsLoss → single logit
        )

        # Store attention for visualisation (detached, no grad)
        self.last_attention_map: torch.Tensor | None = None

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224)

        Returns:
            logits: (B, 1)  — raw logits (apply sigmoid for probability)
        """
        # Spatial stream
        spatial = self.backbone.forward_features(x)   # (B, 1280, 7, 7)

        # Frequency stream
        freq = self.fge(x)                            # (B,  256, 7, 7)

        # Align spatial dimensions if backbone changes resolution
        if spatial.shape[-2:] != freq.shape[-2:]:
            freq = nn.functional.interpolate(
                freq, size=spatial.shape[-2:], mode="bilinear", align_corners=False
            )

        # Cross-consistency attention
        refined, attn = self.cca(spatial, freq)       # (B,1280,7,7), (B,1,7,7)
        self.last_attention_map = attn.detach()

        # Fusion → features
        features = self.aff(refined)                  # (B, 1280)

        # Classify
        logits = self.head(features)                  # (B, 1)
        return logits

    # ------------------------------------------------------------------
    def param_summary(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"Total params: {total:,} | "
            f"Trainable: {trainable:,} ({100*trainable/total:.1f}%)"
        )
