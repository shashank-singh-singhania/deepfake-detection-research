"""
Training loop for FusionDeepfakeDetector — extends src/training/engine.py's
plain classification loop with an auxiliary localization (mask) loss, since
the fusion model's forward(x, return_heatmap=True) also produces a per-pixel
heatmap trained against FF++'s ground-truth manipulation masks.

Evaluation reuses engine.evaluate() unchanged (calls model(x) without
return_heatmap, which returns just the classification logit) — so validation/
test AUC/ACC/EER/AP are computed identically to the Xception/SBI baselines,
keeping every result in the same results table directly comparable.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def train_one_epoch_fusion(model, loader, optimizer, device, scaler=None,
                            cls_weight: float = 1.0, mask_weight: float = 1.0,
                            log_prefix: str = "train", max_grad_norm: float = 1.0):
    """
    `loader` must yield batches with an "image", "label", AND "mask" key —
    i.e. built via src.data.dataset.build_dataloaders(..., return_mask=True).
    Real-sample masks are all-zero by construction (src/data/dataset.py), so
    the mask loss naturally teaches the localization head to predict "nothing
    manipulated" on real frames too, not just to localize on fakes.

    max_grad_norm: gradient clipping threshold applied before optimizer step.
    Critical when mask_weight > 1.0 — without clipping, larger mask gradients
    can cause FP16 overflow under AMP, poisoning BatchNorm running stats and
    causing a NaN cascade that crashes training irreversibly.
    """
    model.train()
    cls_criterion = nn.BCEWithLogitsLoss()
    total_loss, total_cls_loss, total_mask_loss = 0.0, 0.0, 0.0
    n_samples = 0

    pbar = tqdm(loader, desc=log_prefix, leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].float().to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, heatmap = model(images, return_heatmap=True)
                cls_loss = cls_criterion(logits, labels)
                with torch.autocast(device_type="cuda", enabled=False):
                    mask_loss = F.binary_cross_entropy(heatmap.float(), masks.float())
                loss = cls_weight * cls_loss + mask_weight * mask_loss

            # NaN guard: skip this batch entirely if loss is NaN/Inf
            if not torch.isfinite(loss):
                pbar.set_postfix(loss="NaN-SKIP")
                scaler.update()  # still update scale factor (it will reduce it)
                continue

            scaler.scale(loss).backward()
            # Unscale before clipping so max_grad_norm is in real gradient space
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, heatmap = model(images, return_heatmap=True)
            cls_loss = cls_criterion(logits, labels)
            mask_loss = F.binary_cross_entropy(heatmap.float(), masks.float())
            loss = cls_weight * cls_loss + mask_weight * mask_loss
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_cls_loss += cls_loss.item() * bs
        total_mask_loss += mask_loss.item() * bs
        n_samples += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}", cls=f"{cls_loss.item():.4f}", mask=f"{mask_loss.item():.4f}")

    n_samples = max(n_samples, 1)
    return {
        "loss": total_loss / n_samples,
        "cls_loss": total_cls_loss / n_samples,
        "mask_loss": total_mask_loss / n_samples,
    }


@torch.no_grad()
def evaluate_with_localization(model, loader, device, log_prefix: str = "eval"):
    """
    Like engine.evaluate(), but also computes pointing-game accuracy and mean
    IoU on samples that have a non-empty ground-truth mask (fake samples only
    — see src/evaluation/metrics.py). Requires return_mask=True in the loader.
    """
    from src.evaluation.metrics import compute_metrics, compute_pointing_game_accuracy, compute_mask_iou

    model.eval()
    all_labels, all_probs, all_heatmaps, all_masks = [], [], [], []

    for batch in tqdm(loader, desc=log_prefix, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].numpy()
        masks = batch["mask"].numpy()

        logits, heatmap = model(images, return_heatmap=True)
        probs = torch.sigmoid(logits).cpu().numpy()
        heatmap = heatmap.cpu().numpy()

        all_labels.extend(labels.tolist())
        all_probs.extend(probs.tolist())
        all_heatmaps.extend(list(heatmap))
        all_masks.extend(list(masks))

    metrics = compute_metrics(all_labels, all_probs)
    metrics["pointing_game_acc"] = compute_pointing_game_accuracy(all_heatmaps, all_masks)
    metrics["mask_iou"] = compute_mask_iou(all_heatmaps, all_masks)
    return metrics
