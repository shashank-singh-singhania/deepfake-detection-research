"""
Tests for src/training/checkpoint.py — Phase 5. Validates save/load round
trip (model + optimizer + scheduler state all restored correctly) and
scheduler construction/stepping for all three modes.

Run from the repo root:
    python tests/test_checkpoint.py
"""
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SCRATCH = Path(__file__).resolve().parent / "_scratch_checkpoint"


def _tiny_model():
    return torch.nn.Linear(4, 1)


def test_scheduler_construction_and_stepping():
    from src.training.checkpoint import build_scheduler, step_scheduler

    model = _tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    none_sched = build_scheduler("none", opt, epochs=10)
    assert none_sched is None
    step_scheduler(none_sched, "none")  # must not raise

    cosine_sched = build_scheduler("cosine", opt, epochs=10)
    lr_before = opt.param_groups[0]["lr"]
    for _ in range(5):
        opt.step()  # PyTorch warns if scheduler.step() precedes any optimizer.step()
        step_scheduler(cosine_sched, "cosine")
    lr_after = opt.param_groups[0]["lr"]
    assert lr_after != lr_before, "cosine schedule should have changed the LR after 5 steps"
    print(f"[PASS] cosine scheduler steps correctly (lr {lr_before:.2e} -> {lr_after:.2e})")

    opt2 = torch.optim.AdamW(model.parameters(), lr=1e-3)
    plateau_sched = build_scheduler("plateau", opt2, epochs=10)
    lr_before2 = opt2.param_groups[0]["lr"]
    # feed a flat/non-improving metric for more than `patience` steps to force a reduction
    for val in [0.5, 0.5, 0.5, 0.5, 0.5]:
        step_scheduler(plateau_sched, "plateau", val_auc=val)
    lr_after2 = opt2.param_groups[0]["lr"]
    assert lr_after2 < lr_before2, "plateau schedule should reduce LR after repeated non-improvement"
    print(f"[PASS] plateau scheduler reduces LR on stagnation (lr {lr_before2:.2e} -> {lr_after2:.2e})")


def test_checkpoint_save_load_roundtrip():
    from src.training.checkpoint import save_full_checkpoint, load_full_checkpoint, build_scheduler

    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True)
    ckpt_path = SCRATCH / "latest_checkpoint.pt"

    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = build_scheduler("cosine", optimizer, epochs=10)

    # take a few optimizer/scheduler steps so their internal state is non-trivial
    x = torch.randn(8, 4)
    y = torch.randn(8, 1)
    for _ in range(3):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        scheduler.step()

    history = [{"epoch": 0, "train_loss": 0.5}, {"epoch": 1, "train_loss": 0.3}]
    save_full_checkpoint(ckpt_path, model, optimizer, scheduler, epoch=1, best_val_auc=0.83, history=history)

    # build FRESH model/optimizer/scheduler and confirm loading restores everything
    model2 = _tiny_model()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    scheduler2 = build_scheduler("cosine", optimizer2, epochs=10)

    # sanity: before loading, model2's weights differ from model's (different random init)
    assert not torch.allclose(model.weight, model2.weight)

    state = load_full_checkpoint(ckpt_path, model2, optimizer2, scheduler2, device="cpu")

    assert torch.allclose(model.weight, model2.weight), "model weights should match exactly after load"
    assert torch.allclose(model.bias, model2.bias)
    assert optimizer2.state_dict()["param_groups"][0]["lr"] == optimizer.state_dict()["param_groups"][0]["lr"]
    assert scheduler2.state_dict()["last_epoch"] == scheduler.state_dict()["last_epoch"]
    assert state["start_epoch"] == 2, f"expected start_epoch=2 (epoch+1), got {state['start_epoch']}"
    assert state["best_val_auc"] == 0.83
    assert state["history"] == history
    print("[PASS] checkpoint save/load round trip: model, optimizer, scheduler, and metadata all restored correctly")

    shutil.rmtree(SCRATCH, ignore_errors=True)


if __name__ == "__main__":
    test_scheduler_construction_and_stepping()
    test_checkpoint_save_load_roundtrip()
    print("\nALL CHECKPOINT TESTS PASSED")
