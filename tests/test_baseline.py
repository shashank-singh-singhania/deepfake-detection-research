"""
Synthetic smoke test for Phase 3b: baseline model + metrics + training engine.
Validates forward-pass shapes, metric correctness on known cases, and that a
full mini train/eval loop runs end to end on tiny synthetic data (CPU, a
couple of steps) without error — before ever touching real FF++ data or a GPU.

Run from the repo root:
    python tests/test_baseline.py
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRATCH = REPO_ROOT / "tests" / "_scratch_baseline"


def test_metrics_known_cases():
    from src.evaluation.metrics import compute_metrics, compute_eer

    # perfect separation -> AUC should be 1.0, EER should be 0.0
    labels = [0, 0, 0, 1, 1, 1]
    probs = [0.1, 0.05, 0.2, 0.9, 0.95, 0.8]
    m = compute_metrics(labels, probs)
    assert abs(m["auc"] - 1.0) < 1e-6, m
    assert m["eer"] < 1e-6, m
    assert m["acc"] == 1.0, m
    print("[PASS] metrics correct on perfectly-separable synthetic case")

    # random/chance-level separation -> AUC should be close to 0.5
    rng = np.random.default_rng(0)
    labels2 = rng.integers(0, 2, 500)
    probs2 = rng.random(500)
    m2 = compute_metrics(labels2, probs2)
    assert 0.4 < m2["auc"] < 0.6, m2
    print(f"[PASS] metrics near-chance on random synthetic case (auc={m2['auc']:.3f})")


def test_model_forward_shape():
    from src.models.baseline import build_baseline_model

    model = build_baseline_model("xception", pretrained=False)  # no network download in test
    model.eval()
    x = torch.randn(4, 3, 299, 299)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (4,), f"expected shape (4,), got {logits.shape}"
    print("[PASS] XceptionBinaryClassifier forward pass shape correct")


def build_synthetic_manifest():
    import cv2
    import pandas as pd

    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True)

    rows = []
    rng = np.random.default_rng(1)
    for split, n_real, n_fake in [("train", 6, 24), ("val", 4, 16), ("test", 4, 16)]:
        for i in range(n_real):
            p = SCRATCH / f"{split}_real_{i}.jpg"
            cv2.imwrite(str(p), rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
            rows.append({"path": str(p), "label": 0, "method": "youtube", "split": split,
                         "video_id": f"r{i}", "frame_idx": 0, "mask_path": ""})
        for i in range(n_fake):
            p = SCRATCH / f"{split}_fake_{i}.jpg"
            cv2.imwrite(str(p), rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
            rows.append({"path": str(p), "label": 1, "method": "Deepfakes", "split": split,
                         "video_id": f"f{i}", "frame_idx": 0, "mask_path": ""})

    manifest_path = SCRATCH / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def test_mini_training_loop():
    from src.data.dataset import build_dataloaders
    from src.models.baseline import build_baseline_model
    from src.training.engine import train_one_epoch, evaluate

    manifest_path = build_synthetic_manifest()
    loaders = build_dataloaders(str(manifest_path), image_size=64, batch_size=4,
                                 num_workers=0, return_mask=False, balance_train=True)

    model = build_baseline_model("xception", pretrained=False)
    device = "cpu"
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train_loss = train_one_epoch(model, loaders["train"], optimizer, device, scaler=None, log_prefix="test-train")
    assert isinstance(train_loss, float) and train_loss == train_loss, "train loss should be a valid float (not NaN)"

    val_metrics, val_labels, val_probs = evaluate(model, loaders["val"], device, log_prefix="test-val")
    assert set(val_metrics.keys()) == {"auc", "ap", "eer", "acc", "balanced_acc"}
    assert len(val_labels) == 20  # 4 real + 16 fake
    print(f"[PASS] mini training loop ran end-to-end (train_loss={train_loss:.4f}, val_auc={val_metrics['auc']:.3f})")

    shutil.rmtree(SCRATCH, ignore_errors=True)


def test_nan_guard_prevents_weight_corruption():
    """Reproduces (in miniature) the failure mode that corrupted
    fusion_v1_c23's checkpoint on the real DGX run: a batch that produces a
    NaN loss. Confirms train_one_epoch's NaN guard skips that batch's
    optimizer step entirely rather than letting NaN propagate into the
    model's weights."""
    import io
    import contextlib
    from src.data.dataset import build_dataloaders
    from src.models.baseline import build_baseline_model
    from src.training.engine import train_one_epoch

    manifest_path = build_synthetic_manifest()
    loaders = build_dataloaders(str(manifest_path), image_size=64, batch_size=4,
                                 num_workers=0, return_mask=False, balance_train=True)

    model = build_baseline_model("xception", pretrained=False)
    device = "cpu"
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # force the FIRST forward pass of the epoch to return NaN logits, simulating
    # an unstable batch, then let subsequent batches behave normally
    original_forward = model.forward
    call_count = {"n": 0}

    def flaky_forward(x):
        call_count["n"] += 1
        if call_count["n"] == 1:
            out = original_forward(x)
            return out * float("nan")
        return original_forward(x)

    model.forward = flaky_forward

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        train_loss = train_one_epoch(model, loaders["train"], optimizer, device, scaler=None,
                                      log_prefix="test-train-nan")

    assert "[WARN]" in captured.getvalue() and "non-finite loss" in captured.getvalue(), \
        "expected a NaN-skip warning to be printed"
    assert train_loss == train_loss, "epoch-average train_loss should still be a valid float, not NaN"

    all_finite = all(torch.isfinite(p).all() for p in model.parameters())
    assert all_finite, "model weights must remain finite after a NaN batch — this is the exact corruption seen in fusion_v1_c23"
    print(f"[PASS] NaN guard prevented weight corruption from an injected non-finite-loss batch "
          f"(train_loss={train_loss:.4f}, all weights finite)")

    shutil.rmtree(SCRATCH, ignore_errors=True)


if __name__ == "__main__":
    test_metrics_known_cases()
    test_model_forward_shape()
    test_mini_training_loop()
    test_nan_guard_prevents_weight_corruption()
    print("\nALL BASELINE TESTS PASSED")
