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


def train_one_epoch(model, loader, optimizer, device, scaler=None, criterion=None,
                     log_prefix="train", grad_clip_norm: float = 1.0):
    """
    grad_clip_norm: max gradient norm (clip_grad_norm_). Added after a real
    training crash (fusion_v1_c23's latest_checkpoint.pt ended up with NaN
    weights around epoch 12, under AMP with a partially-frozen CLIP
    transformer + custom CNN branches — a known-fragile combination). If a
    batch's loss is NaN/Inf, that batch's optimizer step is skipped entirely
    (weights are never updated from garbage gradients) rather than silently
    corrupting the model; the count of skipped batches is printed once at the
    end of the epoch so a recurring problem is visible, not silent.
    """
    model.train()
    criterion = criterion or nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n_samples = 0
    n_skipped_nan = 0

    pbar = tqdm(loader, desc=log_prefix, leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].float().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                n_skipped_nan += 1
                continue  # skip backward/step entirely — do not let NaN reach the weights

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                n_skipped_nan += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        n_samples += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    if n_skipped_nan > 0:
        print(f"[WARN] {log_prefix}: skipped {n_skipped_nan} batch(es) with non-finite loss "
              f"(optimizer step NOT taken for these — weights were not updated from them)")

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
