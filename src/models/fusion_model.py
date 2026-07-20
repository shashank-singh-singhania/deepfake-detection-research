"""
The novel fusion architecture — Phase 4.

Targets the gap identified in docs/literature_review_deepfake.xlsx (Gap
Analysis tab): most 2024-2025 SOTA papers pick ONE axis (CLIP/VLM semantics,
OR frequency-forensics, OR temporal consistency) and rarely fuse them with a
rigorously validated, quantitative explainability output. This model fuses:

  1. CLIPSemanticBranch — a frozen (mostly) CLIP ViT visual encoder, with only
     the last few transformer blocks + a small adapter trainable. Contributes
     broad semantic/contextual understanding.
  2. FrequencyForensicsBranch — fixed SRM (Steganalysis Rich Model) high-pass
     filters feeding a small trainable CNN, producing a spatial forensic
     feature map. Contributes low-level compression/blending-artifact cues.
  3. CompressionGate — a learned scalar gate, driven by a Laplacian-energy
     proxy for how much high-frequency detail survives in a given image, that
     down-weights the frequency branch's contribution when the input looks
     heavily compressed (frequency artifacts are known to degrade under
     compression — see Gap Analysis #1). This is what makes the frequency
     branch "compression-aware" rather than a fixed-weight fusion.
  4. LocalizationHead — a small conv head on the (gated) frequency spatial
     map, producing a per-pixel manipulation-probability heatmap. Trained
     against FF++'s actual ground-truth manipulation masks (extracted by
     scripts/02_run_preprocessing.py --extract_masks), so explainability can
     be evaluated quantitatively (pointing-game / IoU) rather than only
     qualitatively (Grad-CAM), per Gap Analysis #3.

Optional: TemporalAttentionHead is provided standalone for later use once a
clip-level (multi-frame) dataset variant exists — not yet wired into the main
forward path, since the current Dataset/DataLoader (src/data/dataset.py) is
frame-level. See PROJECT_STRUCTURE.md phase checklist.

Design choice made for tractability: the CLIP branch contributes a pooled
global semantic vector (not spatial tokens) to the fusion; the localization
head operates on the frequency branch's spatial map, which retains spatial
resolution throughout. This keeps the architecture simple enough to train and
debug on a single A100 while still combining both signal types requested by
the identified gap.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import open_clip
    _HAS_OPEN_CLIP = True
except ImportError:
    _HAS_OPEN_CLIP = False


def _rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    """x: (B,3,H,W) -> (B,1,H,W) luminance. Works fine on already-normalized
    (ImageNet mean/std) tensors — this is a fixed linear op, not intensity-
    calibrated, so it's a proxy signal rather than a literal grayscale value."""
    weights = torch.tensor([0.299, 0.587, 0.114], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * weights).sum(dim=1, keepdim=True)


def _build_srm_kernels() -> torch.Tensor:
    """3 standard SRM (Steganalysis Rich Model) high-pass kernels, commonly
    used in image-forensics literature for exposing manipulation/blending
    residuals. Returns a (3, 1, 5, 5) fixed conv weight tensor."""
    k1 = torch.tensor([
        [0, 0, 0, 0, 0],
        [0, -1, 2, -1, 0],
        [0, 2, -4, 2, 0],
        [0, -1, 2, -1, 0],
        [0, 0, 0, 0, 0],
    ], dtype=torch.float32) / 4.0

    k2 = torch.tensor([
        [-1, 2, -2, 2, -1],
        [2, -6, 8, -6, 2],
        [-2, 8, -12, 8, -2],
        [2, -6, 8, -6, 2],
        [-1, 2, -2, 2, -1],
    ], dtype=torch.float32) / 12.0

    k3 = torch.tensor([
        [0, 0, 0, 0, 0],
        [0, 1, -2, 1, 0],
        [0, -2, 4, -2, 0],
        [0, 1, -2, 1, 0],
        [0, 0, 0, 0, 0],
    ], dtype=torch.float32) / 4.0

    kernels = torch.stack([k1, k2, k3], dim=0).unsqueeze(1)  # (3, 1, 5, 5)
    return kernels


class FixedSRMConv(nn.Module):
    """Non-trainable SRM high-pass filter bank applied to the grayscale input."""

    def __init__(self):
        super().__init__()
        self.register_buffer("weight", _build_srm_kernels())

    def forward(self, x_gray: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x_gray, self.weight, padding=2)


class FrequencyForensicsBranch(nn.Module):
    """SRM residual extraction -> small trainable CNN -> spatial feature map."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.srm = FixedSRMConv()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, out_dim, 3, padding=1), nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = _rgb_to_gray(x)
        residual = self.srm(gray)          # (B, 3, H, W)
        feat = self.cnn(residual)          # (B, out_dim, H/4, W/4)
        return feat


class CompressionGate(nn.Module):
    """
    Learned scalar gate in [0,1] driven by a cheap Laplacian-energy proxy for
    how much high-frequency detail survives in the input (heavier compression
    -> less surviving high-frequency energy -> lower gate -> frequency branch
    contributes less to the fused decision). The proxy itself is a fixed,
    non-trainable computation; only the small MLP mapping proxy -> gate is
    learned, so it can calibrate the right down-weighting curve from data
    rather than using a hand-picked threshold.
    """

    def __init__(self):
        super().__init__()
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("laplacian_kernel", laplacian)
        self.mlp = nn.Sequential(nn.Linear(1, 8), nn.ReLU(inplace=True), nn.Linear(8, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = _rgb_to_gray(x)
        lap = F.conv2d(gray, self.laplacian_kernel, padding=1)
        energy = lap.pow(2).mean(dim=[1, 2, 3]).unsqueeze(1)  # (B, 1)
        energy = torch.log1p(energy)  # keep MLP input well-conditioned across compression levels
        gate = torch.sigmoid(self.mlp(energy))  # (B, 1)
        return gate


class CLIPSemanticBranch(nn.Module):
    """
    Frozen (mostly) CLIP visual encoder + small trainable adapter.
    Only the last `n_unfrozen_blocks` transformer blocks (+ final proj/ln_post)
    are trainable — this is the "lightly-adapted" part of "frozen/lightly-
    adapted CLIP" from the project's stated research gap, kept simple
    (partial fine-tuning) rather than a full LoRA implementation for now.
    """

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai",
                 n_unfrozen_blocks: int = 2, proj_dim: int = 256):
        super().__init__()
        if not _HAS_OPEN_CLIP:
            raise ImportError("open_clip_torch is required for CLIPSemanticBranch — pip install open_clip_torch")

        clip_model = open_clip.create_model(model_name, pretrained=pretrained)
        self.visual = clip_model.visual
        del clip_model  # only need the visual tower; drop the text tower to save memory

        self._freeze_except_last_n_blocks(n_unfrozen_blocks)

        clip_dim = self.visual.output_dim
        self.adapter = nn.Linear(clip_dim, proj_dim)
        self.out_dim = proj_dim

    def _freeze_except_last_n_blocks(self, n: int):
        for p in self.visual.parameters():
            p.requires_grad = False
        if n <= 0:
            return
        blocks = self.visual.transformer.resblocks
        for blk in blocks[-n:]:
            for p in blk.parameters():
                p.requires_grad = True
        for attr_name in ["ln_post", "proj"]:
            attr = getattr(self.visual, attr_name, None)
            if attr is None:
                continue
            if isinstance(attr, nn.Parameter):
                attr.requires_grad = True
            else:
                for p in attr.parameters():
                    p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.visual(x)       # (B, clip_dim)
        feat = self.adapter(pooled)   # (B, proj_dim)
        return feat


class LocalizationHead(nn.Module):
    """Small conv head producing a per-pixel manipulation-probability heatmap
    from a spatial feature map, upsampled to the input image resolution."""

    def __init__(self, in_channels: int, hidden: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, spatial_feat: torch.Tensor, out_size) -> torch.Tensor:
        heat = self.conv(spatial_feat)                                   # (B, 1, H', W')
        heat = F.interpolate(heat, size=out_size, mode="bilinear", align_corners=False)
        heat = torch.sigmoid(heat)
        return heat.squeeze(1)                                           # (B, H, W)


class TemporalAttentionHead(nn.Module):
    """
    Standalone self-attention aggregator over a short window of per-frame
    fused feature vectors: (B, T, D) -> (B, D). NOT YET WIRED into
    FusionDeepfakeDetector's forward path, since src/data/dataset.py is
    frame-level, not clip-level. Kept here, tested independently, ready to
    plug in once a clip-level Dataset variant exists (see PROJECT_STRUCTURE.md).
    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(x, x, x)
        x = self.norm(x + attended)
        return x.mean(dim=1)  # (B, D)


class FusionDeepfakeDetector(nn.Module):
    """
    Main model. `forward(x)` returns just the classification logit (shape
    (B,)) so it's a drop-in replacement for XceptionBinaryClassifier and
    works unchanged with src/training/engine.py's train_one_epoch()/evaluate().
    Pass `return_heatmap=True` to also get the localization heatmap (used by
    the dedicated fusion training script for the auxiliary mask loss).
    """

    def __init__(self, clip_model_name: str = "ViT-B-32", clip_pretrained: str = "openai",
                 n_unfrozen_clip_blocks: int = 2, clip_proj_dim: int = 256,
                 freq_feat_dim: int = 128, fusion_hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.clip_branch = CLIPSemanticBranch(clip_model_name, clip_pretrained,
                                               n_unfrozen_clip_blocks, clip_proj_dim)
        self.freq_branch = FrequencyForensicsBranch(out_dim=freq_feat_dim)
        self.compression_gate = CompressionGate()

        fused_dim = clip_proj_dim + freq_feat_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(fusion_hidden_dim, 1)
        self.localization_head = LocalizationHead(in_channels=freq_feat_dim)

    def forward(self, x: torch.Tensor, return_heatmap: bool = False):
        clip_feat = self.clip_branch(x)                     # (B, clip_proj_dim)
        freq_spatial = self.freq_branch(x)                  # (B, freq_feat_dim, H', W')
        gate = self.compression_gate(x)                     # (B, 1)
        gated_freq_spatial = freq_spatial * gate.view(-1, 1, 1, 1)
        pooled_freq = F.adaptive_avg_pool2d(gated_freq_spatial, 1).flatten(1)  # (B, freq_feat_dim)

        fused = torch.cat([clip_feat, pooled_freq], dim=1)
        fused = self.fusion_mlp(fused)
        logit = self.classifier(fused).squeeze(-1)          # (B,)

        if return_heatmap:
            heatmap = self.localization_head(gated_freq_spatial, out_size=x.shape[-2:])
            return logit, heatmap
        return logit

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_fusion_model(**kwargs) -> FusionDeepfakeDetector:
    return FusionDeepfakeDetector(**kwargs)
