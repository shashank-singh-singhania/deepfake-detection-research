"""
Checkpoint save/resume utilities — Phase 5.

Two separate checkpoint files are kept, deliberately, so
scripts/07_run_evaluation.py's simple `torch.load(path) -> state_dict`
loading never needs to change:

  - best_model.pt        : model.state_dict() ONLY, saved whenever val AUC
                            improves. This is what src/evaluation/evaluate.py
                            and scripts/07 load for evaluation.
  - latest_checkpoint.pt : full training state (model + optimizer + scheduler
                            + epoch + history + best_val_auc), OVERWRITTEN
                            every epoch. Used only by --resume_from_checkpoint
                            to continue an interrupted run — never used for
                            evaluation.
"""
from pathlib import Path
from typing import Optional

import torch


def save_full_checkpoint(path, model, optimizer, scheduler, epoch: int,
                          best_val_auc: float, history: list):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_val_auc": best_val_auc,
        "history": history,
    }, path)


def load_full_checkpoint(path, model, optimizer, scheduler, device) -> dict:
    """Restores model/optimizer/scheduler in place. Returns a dict with
    start_epoch (the NEXT epoch to run), best_val_auc, and history so the
    training script's loop and early-stopping logic can pick up where they
    left off."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return {
        "start_epoch": ckpt["epoch"] + 1,
        "best_val_auc": ckpt["best_val_auc"],
        "history": ckpt["history"],
    }


def build_scheduler(name: str, optimizer, epochs: int):
    """name: 'none' | 'cosine' | 'plateau'.
    'plateau' must be stepped with scheduler.step(val_auc) (mode='max');
    'cosine' and 'none' are stepped with scheduler.step() (or not stepped, for 'none')."""
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    raise ValueError(f"Unknown lr_scheduler '{name}'. Choices: none, cosine, plateau")


def step_scheduler(scheduler, name: str, val_auc: Optional[float] = None):
    if scheduler is None:
        return
    if name == "plateau":
        scheduler.step(val_auc)
    else:
        scheduler.step()
