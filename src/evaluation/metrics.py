"""
Standard deepfake-detection metrics, used consistently across baseline
reproduction, the novel model, and every evaluation protocol later (in-dataset,
cross-manipulation, cross-dataset). Keeping this centralized means every
number in the paper's results tables comes from the same computation.
"""
import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score, average_precision_score, balanced_accuracy_score,
)


def compute_eer(labels: np.ndarray, probs: np.ndarray) -> float:
    """Equal Error Rate: point where false-positive rate == false-negative rate."""
    fpr, tpr, _ = roc_curve(labels, probs)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2)


def compute_metrics(labels, probs, threshold: float = 0.5) -> dict:
    """
    labels: 0/1 ground truth (real/fake)
    probs: predicted probability of being FAKE (i.e. sigmoid output, class 1)

    Includes BOTH `acc` (raw accuracy) and `balanced_acc` (average of
    per-class recall). Raw accuracy is sensitive to the real:fake ratio of
    whatever subset you evaluate on — e.g. an "all methods combined" FF++
    test split is naturally ~1:4 real:fake (4 manipulation methods per real
    video), while a "per-method" subset (this method's fakes + ALL reals) is
    closer to 1:1. A model with a miscalibrated decision threshold can show
    wildly different `acc` across these two views even though its underlying
    ranking (AUC) barely changes — `balanced_acc` is threshold-sensitive too,
    but at least isn't distorted by the subset's class ratio, so prefer it
    when comparing accuracy across differently-composed evaluation subsets.
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    preds = (probs >= threshold).astype(int)

    metrics = {}
    if len(np.unique(labels)) < 2:
        # AUC/EER undefined with only one class present (e.g. tiny debug batch)
        metrics["auc"] = float("nan")
        metrics["eer"] = float("nan")
        metrics["ap"] = float("nan")
        metrics["balanced_acc"] = float("nan")
    else:
        metrics["auc"] = float(roc_auc_score(labels, probs))
        metrics["ap"] = float(average_precision_score(labels, probs))
        metrics["eer"] = compute_eer(labels, probs)
        metrics["balanced_acc"] = float(balanced_accuracy_score(labels, preds))
    metrics["acc"] = float(accuracy_score(labels, preds))
    return metrics


def compute_pointing_game_accuracy(heatmaps, masks) -> float:
    """
    Quantitative explainability metric (used instead of only qualitative
    Grad-CAM, per the project's identified research gap): for each FAKE
    sample with a non-empty ground-truth manipulation mask, checks whether
    the predicted heatmap's single highest-activation pixel falls inside the
    ground-truth manipulated region. Real / no-mask samples are skipped
    (there's no manipulation region to point at).

    heatmaps, masks: iterable of (H, W) arrays (numpy or tensors), same shape
    per pair. masks should be in [0,1] (mask > 0.5 treated as "inside").
    """
    correct, total = 0, 0
    for hm, m in zip(heatmaps, masks):
        hm = np.asarray(hm)
        m = np.asarray(m)
        if m.sum() <= 0:
            continue
        total += 1
        peak_idx = np.unravel_index(np.argmax(hm), hm.shape)
        if m[peak_idx] > 0.5:
            correct += 1
    return correct / total if total > 0 else float("nan")


def compute_mask_iou(heatmaps, masks, heat_threshold: float = 0.5, mask_threshold: float = 0.5) -> float:
    """
    Mean IoU between thresholded predicted heatmaps and ground-truth masks,
    over FAKE samples with a non-empty mask only (see compute_pointing_game_accuracy).
    """
    ious = []
    for hm, m in zip(heatmaps, masks):
        hm = np.asarray(hm)
        m = np.asarray(m)
        if m.sum() <= 0:
            continue
        pred = hm >= heat_threshold
        gt = m >= mask_threshold
        union = np.logical_or(pred, gt).sum()
        if union == 0:
            continue
        inter = np.logical_and(pred, gt).sum()
        ious.append(inter / union)
    return float(np.mean(ious)) if ious else float("nan")
