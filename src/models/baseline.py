"""
Baseline deepfake classifier models — Phase 3b.

Starts with Xception, the original FF++ paper's baseline architecture, since
it's the standard first comparison point almost every later paper reports
against. SBI (Self-Blended Images) training is a separate augmentation
strategy layered on top of a backbone like this one — see
docs/literature_review_deepfake.xlsx (Gap Analysis) and PROJECT_STRUCTURE.md
for where that fits in next.
"""
import torch
import torch.nn as nn

try:
    import timm
    _HAS_TIMM = True
except ImportError:
    _HAS_TIMM = False


class XceptionBinaryClassifier(nn.Module):
    """
    Xception backbone (ImageNet-pretrained) + a single binary logit head.
    Outputs a raw logit (use torch.sigmoid for probability, or BCEWithLogitsLoss
    directly during training — do not double-apply sigmoid).
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        if not _HAS_TIMM:
            raise ImportError("timm is required for XceptionBinaryClassifier — pip install timm")
        # num_classes=0 -> backbone returns pooled features, no classifier head
        self.backbone = timm.create_model("xception", pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feat_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        feats = self.dropout(feats)
        logit = self.classifier(feats).squeeze(-1)
        return logit


MODEL_REGISTRY = {
    "xception": XceptionBinaryClassifier,
}


def build_baseline_model(name: str = "xception", pretrained: bool = True, **kwargs) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown baseline model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](pretrained=pretrained, **kwargs)
