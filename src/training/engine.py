"""
Shared train/eval loop logic — used by scripts/04_train_baseline.py now, and
by the novel fusion model's training script later (Phase 5), so the training
mechanics (AMP, metric computation) stay identical across every model we
compare in the paper.
"""
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.evaluation.metrics import compute_metrics


def train_one_epoch(model, loader, optimizer, device, scaler=None, criterion=None, log_prefix="train"):
    model.train()
    criterion = criterion or nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n_samples = 0

    pbar = tqdm(loader, desc=log_prefix, leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].float().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        n_samples += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate(model, loader, device, log_prefix="eval"):
    model.eval()
    all_labels, all_probs = [], []

    for batch in tqdm(loader, desc=log_prefix, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].numpy()
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.tolist())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    metrics = compute_metrics(all_labels, all_probs)
    return metrics, all_labels, all_probs
