"""
TriConsistencyNet — Evaluation Script

Runs evaluation on test split in manifest.csv.
Prints overall metrics and per-method breakdown (Deepfakes, Face2Face, FaceSwap, NeuralTextures).

Usage:
  python another_model/evaluate.py \
    --checkpoint experiments/triconsistencynet_c23/best_model.pt \
    --manifest data/processed/manifest.csv
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure local src import works
ANOTHER_MODEL_DIR = Path(__file__).resolve().parent
SRC_DIR = ANOTHER_MODEL_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import TriConsistencyNet
from dataset import FFPPDataset


def compute_metrics(probs: np.ndarray, targets: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

    if len(np.unique(targets)) < 2:
        return {"auc": 0.5, "ap": 0.5, "eer": 0.5, "balanced_acc": 0.5, "acc": 0.5}

    auc = float(roc_auc_score(targets, probs))
    ap = float(average_precision_score(targets, probs))

    preds = (probs >= 0.5).astype(float)
    acc = float(np.mean(preds == targets))

    real_mask = (targets == 0)
    fake_mask = (targets == 1)
    acc_real = float(np.mean(preds[real_mask] == 0)) if np.sum(real_mask) > 0 else 0.0
    acc_fake = float(np.mean(preds[fake_mask] == 1)) if np.sum(fake_mask) > 0 else 0.0
    balanced_acc = (acc_real + acc_fake) / 2.0

    fpr, tpr, _ = roc_curve(targets, probs, pos_label=1)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float(fpr[eer_idx])

    return {
        "auc": auc,
        "ap": ap,
        "eer": eer,
        "balanced_acc": balanced_acc,
        "acc": acc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--manifest", type=str, default="data/processed/manifest.csv")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Checkpoint: {args.checkpoint}")

    model = TriConsistencyNet(freeze_backbone=True).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    test_ds = FFPPDataset(args.manifest, split="test", image_size=args.image_size, augment=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    all_probs = []
    all_targets = []
    all_methods = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].numpy()
            methods = batch["method"]

            with autocast():
                logits = model(images)

            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            all_probs.extend(probs)
            all_targets.extend(labels)
            all_methods.extend(methods)

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    all_methods = np.array(all_methods)

    print("\n=== In-dataset (test split) ===")
    overall_metrics = compute_metrics(all_probs, all_targets)
    print(f"All methods combined: {overall_metrics}")

    print("Per-method breakdown:")
    methods_unique = set(all_methods) - {"real", "unknown"}
    for m in sorted(methods_unique):
        m_mask = (all_methods == m) | (all_methods == "real")
        m_probs = all_probs[m_mask]
        m_targets = all_targets[m_mask]
        m_metrics = compute_metrics(m_probs, m_targets)
        print(f"  {m}: {m_metrics}")


if __name__ == "__main__":
    main()
