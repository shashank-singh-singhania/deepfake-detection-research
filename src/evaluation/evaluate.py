"""
Phase 6 — Multi-protocol evaluation suite.

Runs ONE inference pass over a manifest's test split and derives multiple
evaluation protocols from it (grouping by the "method" field every
FFPPDataset sample already carries), rather than needing separate dataloaders
per protocol:

  A. In-dataset       — full test split, all methods combined (the headline
                         number most papers report, and the least informative
                         on its own — see Gap Analysis #7 in
                         docs/literature_review_deepfake.xlsx)
  B. Per-method        — test split broken down by manipulation method
                         (Deepfakes / Face2Face / FaceSwap / NeuralTextures
                         each vs. real). If a checkpoint was trained with a
                         method excluded (see --exclude_methods on
                         scripts/04 and scripts/06), a noticeably lower score
                         on that method's row here IS the cross-manipulation
                         leave-one-out generalization result.
  C. Cross-dataset zero-shot — pass an entirely different manifest.csv (e.g.
                         built by re-running scripts/02 pointed at a Celeb-DF
                         copy — not yet built, see PROJECT_STRUCTURE.md) via
                         --cross_dataset_manifest, evaluated with NO
                         fine-tuning.
  D. Explainability (fusion architecture only) — pointing-game accuracy +
                         mask IoU against FF++ ground-truth masks.

This module does the computation; scripts/07_run_evaluation.py is the CLI.
"""
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import FFPPDataset
from src.evaluation.metrics import (
    compute_metrics, compute_pointing_game_accuracy, compute_mask_iou,
    compute_heatmap_stats, compute_best_threshold_iou,
)
from src.models.baseline import build_baseline_model
from src.models.fusion_model import build_fusion_model

ARCHITECTURES = ("xception", "fusion")


def load_model(architecture: str, checkpoint_path: str, device: str, **model_kwargs):
    if architecture == "xception":
        model = build_baseline_model("xception", pretrained=False)
    elif architecture == "fusion":
        model = build_fusion_model(**model_kwargs)
    else:
        raise ValueError(f"Unknown architecture '{architecture}'. Choices: {ARCHITECTURES}")

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def run_inference(model, architecture: str, manifest_csv: str, split: str, image_size: int,
                   batch_size: int, device: str, num_workers: int = 8) -> list:
    """Single inference pass. Returns a list of per-sample record dicts —
    grouping/aggregation into protocols happens afterward in summarize_protocols()."""
    return_mask = (architecture == "fusion")
    ds = FFPPDataset(manifest_csv, split=split, image_size=image_size, return_mask=return_mask)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    records = []
    for batch in tqdm(loader, desc=f"eval[{split}]", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].numpy()
        methods = batch["method"]

        if architecture == "fusion":
            logits, heatmaps = model(images, return_heatmap=True)
            heatmaps = heatmaps.cpu().numpy()
            masks = batch["mask"].numpy()
        else:
            logits = model(images)
            heatmaps = [None] * len(labels)
            masks = [None] * len(labels)

        probs = torch.sigmoid(logits).cpu().numpy()
        for i in range(len(labels)):
            records.append({
                "label": int(labels[i]),
                "prob": float(probs[i]),
                "method": methods[i],
                "heatmap": heatmaps[i] if architecture == "fusion" else None,
                "mask": masks[i] if architecture == "fusion" else None,
            })
    return records


def summarize_protocols(records: list) -> dict:
    """Derives protocols A, B, (D) from a single set of inference records.
    Protocol C (cross-dataset) is just this function called again on a
    different manifest's records — see scripts/07_run_evaluation.py."""
    results = {}

    labels_all = [r["label"] for r in records]
    probs_all = [r["prob"] for r in records]
    results["in_dataset_all_methods"] = compute_metrics(labels_all, probs_all)

    fake_methods = sorted({r["method"] for r in records if r["label"] == 1})
    per_method = {}
    for m in fake_methods:
        sub = [r for r in records if r["method"] in (m, "youtube")]
        if len(sub) == 0:
            continue
        per_method[m] = compute_metrics([r["label"] for r in sub], [r["prob"] for r in sub])
    results["per_method"] = per_method

    has_masks = any(r.get("mask") is not None for r in records)
    if has_masks:
        heatmaps = [r["heatmap"] for r in records if r["heatmap"] is not None]
        masks = [r["mask"] for r in records if r["mask"] is not None]
        results["explainability"] = {
            "pointing_game_acc": compute_pointing_game_accuracy(heatmaps, masks),
            "mask_iou": compute_mask_iou(heatmaps, masks),
            "heatmap_stats": compute_heatmap_stats(heatmaps),
            "best_threshold_iou": compute_best_threshold_iou(heatmaps, masks),
        }

    return results


def run_multi_protocol_evaluation(model, architecture: str, manifest_csv: str, image_size: int,
                                   batch_size: int, device: str, num_workers: int = 8,
                                   cross_dataset_manifest: Optional[str] = None,
                                   cross_dataset_split: str = "test") -> dict:
    """Full orchestration: in-dataset test split (protocols A+B[+D]), plus an
    optional cross-dataset manifest (protocol C, same summarization reused)."""
    records = run_inference(model, architecture, manifest_csv, split="test", image_size=image_size,
                             batch_size=batch_size, device=device, num_workers=num_workers)
    results = {"in_dataset": summarize_protocols(records)}

    if cross_dataset_manifest is not None:
        cross_records = run_inference(model, architecture, cross_dataset_manifest, split=cross_dataset_split,
                                       image_size=image_size, batch_size=batch_size, device=device,
                                       num_workers=num_workers)
        results["cross_dataset"] = summarize_protocols(cross_records)

    return results
