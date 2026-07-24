"""
TriConsistencyNet

Complete Architecture

Author: Shashank Singh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from src.models.triconsistencynet.frequency import (
    FrequencyGuidanceEncoder,
)

from src.models.triconsistencynet.attention import (
    CrossConsistencyAttention,
)

from src.models.triconsistencynet.fusion import (
    AdaptiveFeatureFusion,
)

from src.utils.config import ConfigLoader


class TriConsistencyNet(nn.Module):

    def __init__(self):

        super().__init__()

        config = ConfigLoader().load("model.yaml")

        self.backbone = timm.create_model(
            config.model.backbone,
            pretrained=config.model.pretrained,
            num_classes=0,
            global_pool="",
        )

        self.frequency_encoder = FrequencyGuidanceEncoder()

        self.cross_attention = CrossConsistencyAttention()

        self.fusion = AdaptiveFeatureFusion()

        self.dropout = nn.Dropout(
            config.model.dropout
        )

        self.classifier = nn.Linear(
            1280,
            config.model.num_classes,
        )

        self.last_attention_map = None

    def forward(self, x):

        # -------------------------------
        # Spatial Features
        # -------------------------------

        spatial = self.backbone.forward_features(x)

        # -------------------------------
        # Frequency Features
        # -------------------------------

        frequency = self.frequency_encoder(x)

        if spatial.shape[-2:] != frequency.shape[-2:]:

            frequency = F.interpolate(
                frequency,
                size=spatial.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # -------------------------------
        # Cross-Consistency Attention
        # -------------------------------

        refined, attention = self.cross_attention(
            spatial,
            frequency,
        )

        self.last_attention_map = attention

        # -------------------------------
        # Adaptive Fusion
        # -------------------------------

        features = self.fusion(refined)

        # -------------------------------
        # Classification
        # -------------------------------

        features = self.dropout(features)

        logits = self.classifier(features)

        return logits
