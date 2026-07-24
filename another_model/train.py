"""
TriConsistencyNet — Standalone Training Script

Reads from data/processed/manifest.csv.
Saves checkpoints and metrics to experiments/{run_name}/.

Usage:
  python another_model/train.py \
    --manifest data/processed/manifest.csv \
    --epochs 30 --batch_size 64 --lr 1e-4 \
    --run_name triconsistencynet_c23
"""

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Ensure local src import works
ANOTHER_MODEL_DIR = Path(__file__).resolve().parent
SRC_DIR = ANOTHER_MODEL_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import TriConsistencyNet
from dataset import get_loaders


def compute_metrics(all_probs: np.ndarray, all_targets: np.ndarray) -> dict:
    """Computes AUC, Accuracy, and EER."""
    from sklearn.metrics import roc_auc_score, roc_curve

    if len(np.unique(all_targets)) < 2:
        return {"auc": 0.5, "acc": 0.5, "eer": 0.5}

    auc = roc_auc_score(all_targets, all_probs)

    preds = (all_probs >= 0.5).astype(float)
    acc = np.mean(preds == all_targets)

    fpr, tpr, thresholds = roc_curve(all_targets, all_probs, pos_label=1)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float(fpr[eer_idx])

    return {"auc": float(auc), "acc": float(acc), "eer": float(eer)}


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, max_grad_norm=1.0):
    model.train()
    total_loss = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                pbar.set_postfix(loss="NaN-SKIP")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        b = images.size(0)
        total_loss += loss.item() * b
        total_samples += b
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_probs = []
    all_targets = []

    for batch in tqdm(loader, desc="eval", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).unsqueeze(1)

        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        probs = torch.sigmoid(logits).cpu().numpy().ravel()
        targets = labels.cpu().numpy().ravel()

        b = images.size(0)
        total_loss += loss.item() * b
        total_samples += b

        all_probs.extend(probs)
        all_targets.extend(targets)

    avg_loss = total_loss / max(total_samples, 1)
    metrics = compute_metrics(np.array(all_probs), np.array(all_targets))
    metrics["loss"] = avg_loss
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train TriConsistencyNet")
    parser.add_argument("--manifest", type=str, default="data/processed/manifest.csv")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--run_name", type=str, default="triconsistencynet_c23")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Run dir: experiments/{args.run_name}")

    run_dir = Path("experiments") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = get_loaders(
        args.manifest,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=4,
    )
    print(f"train: {len(train_loader.dataset)} samples | val: {len(val_loader.dataset)} samples | test: {len(test_loader.dataset)} samples")

    model = TriConsistencyNet(freeze_backbone=True, dropout=0.3).to(device)
    print(model.param_summary())

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler() if device.type == "cuda" else None

    start_epoch = 0
    best_val_auc = 0.0
    patience_counter = 0

    if args.resume_from_checkpoint and Path(args.resume_from_checkpoint).exists():
        ckpt = torch.load(args.resume_from_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt and scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_auc = ckpt.get("best_val_auc", 0.0)
        print(f"Resumed from {args.resume_from_checkpoint}: starting epoch {start_epoch}, best_val_auc={best_val_auc:.4f}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr_curr = optimizer.param_groups[0]["lr"]

        print(
            f"[epoch {epoch}] train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_auc={val_metrics['auc']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} val_eer={val_metrics['eer']:.4f} | "
            f"lr={lr_curr:.2e} | {elapsed:.1f}s"
        )

        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_auc": best_val_auc,
            "val_metrics": val_metrics,
        }

        torch.save(checkpoint_state, run_dir / "latest_checkpoint.pt")

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            patience_counter = 0
            torch.save(checkpoint_state, run_dir / "best_model.pt")
            print(f"  -> new best val_auc={best_val_auc:.4f}, saved best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    print(f"Training completed. Best val_auc = {best_val_auc:.4f}")


if __name__ == "__main__":
    main()
