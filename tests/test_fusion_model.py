"""
Tests for the novel fusion architecture (Phase 4). Uses pretrained=None for
the CLIP branch (random init, no network download) so these run anywhere,
including CPU-only, without needing real weights. Validates: individual
branch shapes, compression gate behavior on synthetic high/low-frequency
inputs, full model forward pass (with and without heatmap), gradient flow
into every intended-trainable component, and that frozen CLIP blocks
actually stay frozen.

Run from the repo root:
    python tests/test_fusion_model.py

NOTE: these tests build a real (randomly-initialized) CLIP ViT-B-32, so they
are slower than the other synthetic tests (~10-30s on CPU) and use several
hundred MB of memory. That's expected.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

IMAGE_SIZE = 224  # CLIP ViT-B-32 default positional embedding size


def test_frequency_branch_shape():
    from src.models.fusion_model import FrequencyForensicsBranch
    branch = FrequencyForensicsBranch(out_dim=64)
    x = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    out = branch(x)
    assert out.shape == (2, 64, IMAGE_SIZE // 4, IMAGE_SIZE // 4), out.shape
    print(f"[PASS] FrequencyForensicsBranch output shape correct: {tuple(out.shape)}")


def test_srm_kernels_are_fixed_not_trainable():
    from src.models.fusion_model import FixedSRMConv
    srm = FixedSRMConv()
    assert not srm.weight.requires_grad, "SRM kernels must be fixed (non-trainable) by design"
    assert srm.weight.shape == (3, 1, 5, 5)
    print("[PASS] SRM kernels are fixed (non-trainable), correct shape")


def test_compression_gate_responds_to_frequency_content():
    from src.models.fusion_model import CompressionGate
    gate = CompressionGate()
    gate.eval()

    torch.manual_seed(0)
    smooth = torch.linspace(0, 1, IMAGE_SIZE).view(1, 1, 1, IMAGE_SIZE).expand(1, 3, IMAGE_SIZE, IMAGE_SIZE).clone()
    noisy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    with torch.no_grad():
        g_smooth = gate(smooth).item()
        g_noisy = gate(noisy).item()

    assert 0.0 <= g_smooth <= 1.0 and 0.0 <= g_noisy <= 1.0
    assert abs(g_smooth - g_noisy) > 1e-4, \
        f"gate should respond differently to smooth vs noisy input, got {g_smooth} vs {g_noisy}"
    print(f"[PASS] CompressionGate produces distinct, valid [0,1] outputs "
          f"(smooth={g_smooth:.4f}, noisy={g_noisy:.4f})")


def test_clip_branch_freezing():
    from src.models.fusion_model import CLIPSemanticBranch
    branch = CLIPSemanticBranch(model_name="ViT-B-32", pretrained=None, n_unfrozen_blocks=2, proj_dim=128)

    blocks = branch.visual.transformer.resblocks
    n_blocks = len(blocks)
    for i, blk in enumerate(blocks):
        expected_trainable = (i >= n_blocks - 2)
        actual_trainable = all(p.requires_grad for p in blk.parameters())
        assert actual_trainable == expected_trainable, \
            f"block {i}/{n_blocks}: expected trainable={expected_trainable}, got {actual_trainable}"
    assert all(p.requires_grad for p in branch.adapter.parameters()), "adapter must be trainable"
    print(f"[PASS] CLIPSemanticBranch freezing correct: last 2/{n_blocks} blocks + adapter trainable, rest frozen")


def test_clip_branch_forward_shape():
    from src.models.fusion_model import CLIPSemanticBranch
    branch = CLIPSemanticBranch(model_name="ViT-B-32", pretrained=None, n_unfrozen_blocks=1, proj_dim=256)
    branch.eval()
    x = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        out = branch(x)
    assert out.shape == (2, 256), out.shape
    print(f"[PASS] CLIPSemanticBranch forward shape correct: {tuple(out.shape)}")


def test_full_model_forward_and_gradients():
    from src.models.fusion_model import FusionDeepfakeDetector

    model = FusionDeepfakeDetector(
        clip_model_name="ViT-B-32", clip_pretrained=None, n_unfrozen_clip_blocks=1,
        clip_proj_dim=128, freq_feat_dim=64, fusion_hidden_dim=128, dropout=0.1,
    )
    model.train()
    x = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE, requires_grad=False)
    labels = torch.tensor([0.0, 1.0])

    logit = model(x)
    assert logit.shape == (2,), logit.shape

    loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, labels)
    loss.backward()

    assert model.classifier.weight.grad is not None and model.classifier.weight.grad.abs().sum() > 0
    assert model.freq_branch.cnn[0].weight.grad is not None and model.freq_branch.cnn[0].weight.grad.abs().sum() > 0
    assert model.clip_branch.adapter.weight.grad is not None and model.clip_branch.adapter.weight.grad.abs().sum() > 0
    last_clip_block = model.clip_branch.visual.transformer.resblocks[-1]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in last_clip_block.parameters()), \
        "last unfrozen CLIP block should receive gradient"

    first_clip_block = model.clip_branch.visual.transformer.resblocks[0]
    assert all(p.grad is None for p in first_clip_block.parameters()), \
        "frozen (early) CLIP blocks must not receive gradient"
    assert model.freq_branch.srm.weight.grad is None, "fixed SRM kernels must never receive gradient"
    print("[PASS] full model forward + backward: gradients flow correctly, frozen parts stay frozen")

    model.zero_grad()
    logit2, heatmap = model(x, return_heatmap=True)
    assert logit2.shape == (2,)
    assert heatmap.shape == (2, IMAGE_SIZE, IMAGE_SIZE), heatmap.shape
    assert (heatmap >= 0).all() and (heatmap <= 1).all(), "heatmap must be in [0,1] (sigmoid output)"

    mask = torch.zeros(2, IMAGE_SIZE, IMAGE_SIZE)
    mask[1] = 1.0
    aux_loss = torch.nn.functional.binary_cross_entropy(heatmap, mask)
    aux_loss.backward()
    assert model.localization_head.conv[0].weight.grad is not None and \
        model.localization_head.conv[0].weight.grad.abs().sum() > 0
    print(f"[PASS] heatmap forward shape correct {tuple(heatmap.shape)}, "
          f"in valid [0,1] range, localization head receives gradient from mask loss")


def test_parameter_counts_sane():
    from src.models.fusion_model import FusionDeepfakeDetector
    model = FusionDeepfakeDetector(clip_model_name="ViT-B-32", clip_pretrained=None,
                                    n_unfrozen_clip_blocks=2, clip_proj_dim=128,
                                    freq_feat_dim=64, fusion_hidden_dim=128)
    total = model.total_parameter_count()
    trainable = model.trainable_parameter_count()
    assert trainable < total, "freezing should mean trainable < total parameters"
    frac = trainable / total
    print(f"[PASS] parameter counts sane: total={total:,}, trainable={trainable:,} ({frac:.1%})")


def test_localization_metrics():
    from src.evaluation.metrics import compute_pointing_game_accuracy, compute_mask_iou
    import numpy as np

    # sample 0: perfect prediction (heatmap peak + full overlap inside a 4x4 mask region)
    mask0 = np.zeros((10, 10)); mask0[2:6, 2:6] = 1.0
    heat0 = np.zeros((10, 10)); heat0[2:6, 2:6] = 0.9; heat0[3, 3] = 1.0  # peak inside mask

    # sample 1: wrong prediction (heatmap peak far outside the mask region)
    mask1 = np.zeros((10, 10)); mask1[2:4, 2:4] = 1.0
    heat1 = np.zeros((10, 10)); heat1[8, 8] = 1.0  # peak outside mask, no overlap at all

    # sample 2: real sample, empty mask -> must be skipped by both metrics
    mask2 = np.zeros((10, 10))
    heat2 = np.random.rand(10, 10)

    pg_acc = compute_pointing_game_accuracy([heat0, heat1, heat2], [mask0, mask1, mask2])
    assert abs(pg_acc - 0.5) < 1e-6, f"expected 1/2 correct (sample2 skipped), got {pg_acc}"

    iou = compute_mask_iou([heat0, heat1, heat2], [mask0, mask1, mask2], heat_threshold=0.5, mask_threshold=0.5)
    # sample0: perfect overlap -> iou=1; sample1: zero overlap -> iou=0; sample2 skipped -> mean=0.5
    assert abs(iou - 0.5) < 1e-6, f"expected mean IoU 0.5 ((1+0)/2, sample2 skipped), got {iou}"

    print(f"[PASS] localization metrics correct: pointing_game_acc={pg_acc}, mean_iou={iou} (real/empty-mask samples correctly skipped)")


def test_fusion_training_loop_mini():
    """End-to-end: build tiny synthetic manifest (with masks) -> real
    build_dataloaders(return_mask=True) -> one epoch of train_one_epoch_fusion
    -> evaluate_with_localization. Validates the whole Phase 4 training path,
    not just the model in isolation."""
    import shutil
    import cv2
    import numpy as np
    import pandas as pd
    import torch as _torch
    from torch.optim import AdamW

    scratch = Path(__file__).resolve().parent / "_scratch_fusion_training"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)

    rng = np.random.default_rng(3)
    rows = []
    for split, n_real, n_fake in [("train", 4, 4), ("val", 2, 2), ("test", 2, 2)]:
        for i in range(n_real):
            p = scratch / f"{split}_real_{i}.jpg"
            cv2.imwrite(str(p), rng.integers(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))
            rows.append({"path": str(p), "label": 0, "method": "youtube", "split": split,
                         "video_id": f"r{i}", "frame_idx": 0, "mask_path": ""})
        for i in range(n_fake):
            p = scratch / f"{split}_fake_{i}.jpg"
            mp = scratch / f"{split}_fake_mask_{i}.jpg"
            cv2.imwrite(str(p), rng.integers(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))
            cv2.imwrite(str(mp), rng.integers(0, 255, (IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8))
            rows.append({"path": str(p), "label": 1, "method": "Deepfakes", "split": split,
                         "video_id": f"f{i}", "frame_idx": 0, "mask_path": str(mp)})

    manifest_path = scratch / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    from src.data.dataset import build_dataloaders
    from src.models.fusion_model import FusionDeepfakeDetector
    from src.training.train_fusion import train_one_epoch_fusion, evaluate_with_localization

    loaders = build_dataloaders(str(manifest_path), image_size=IMAGE_SIZE, batch_size=4,
                                 num_workers=0, return_mask=True, balance_train=True)

    model = FusionDeepfakeDetector(clip_model_name="ViT-B-32", clip_pretrained=None,
                                    n_unfrozen_clip_blocks=1, clip_proj_dim=64,
                                    freq_feat_dim=32, fusion_hidden_dim=64)
    device = "cpu"
    model.to(device)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    train_stats = train_one_epoch_fusion(model, loaders["train"], optimizer, device, scaler=None,
                                          mask_weight=1.0, log_prefix="test-fusion-train")
    assert all(v == v for v in train_stats.values()), f"NaN in training stats: {train_stats}"  # v==v is False for NaN

    val_metrics = evaluate_with_localization(model, loaders["val"], device, log_prefix="test-fusion-val")
    expected_keys = {"auc", "ap", "eer", "acc", "pointing_game_acc", "mask_iou"}
    assert expected_keys.issubset(val_metrics.keys()), val_metrics.keys()

    shutil.rmtree(scratch, ignore_errors=True)
    print(f"[PASS] full fusion training loop ran end-to-end "
          f"(train loss={train_stats['loss']:.4f}, val_auc={val_metrics['auc']:.3f}, "
          f"val_pointing_game={val_metrics['pointing_game_acc']})")


if __name__ == "__main__":
    test_frequency_branch_shape()
    test_srm_kernels_are_fixed_not_trainable()
    test_compression_gate_responds_to_frequency_content()
    test_clip_branch_freezing()
    test_clip_branch_forward_shape()
    test_full_model_forward_and_gradients()
    test_parameter_counts_sane()
    test_localization_metrics()
    test_fusion_training_loop_mini()
    print("\nALL FUSION MODEL TESTS PASSED")
