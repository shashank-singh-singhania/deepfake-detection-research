"""
Script 10 — Evaluate Video & Image Compression Robustness.

Evaluates trained checkpoints (Xception vs Fusion v4) under controlled JPEG compression
degradation levels (JPEG Quality Q = 100, 90, 80, 70, 60, 50) simulating real-world
social media re-encoding (YouTube, WhatsApp, Twitter, TikTok).

Usage:
    python scripts/10_evaluate_compression_robustness.py \
        --architecture fusion \
        --checkpoint experiments/fusion_v4_c23/best_model.pt \
        --manifest data/processed/manifest.csv \
        --freq_backbone efficientnet_b0 \
        --output evaluation_results/fusion_v4_compression_robustness.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.evaluation.evaluate import load_model
from src.evaluation.metrics import compute_metrics


def build_compressed_transform(image_size: int, quality: int):
    """Builds evaluation transform with forced JPEG quality degradation."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.ImageCompression(quality_lower=quality, quality_upper=quality, p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--architecture", required=True, choices=["xception", "fusion"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--qualities", nargs="+", type=int, default=[100, 90, 80, 70, 60, 50],
                    help="JPEG quality factors to evaluate")
    ap.add_argument("--output", default="evaluation_results/compression_robustness.json")

    # fusion args
    ap.add_argument("--clip_model_name", default="ViT-B-32")
    ap.add_argument("--clip_pretrained", default="openai")
    ap.add_argument("--n_unfrozen_clip_blocks", type=int, default=2)
    ap.add_argument("--clip_proj_dim", type=int, default=256)
    ap.add_argument("--freq_feat_dim", type=int, default=128)
    ap.add_argument("--freq_backbone", default="simple_cnn", choices=["simple_cnn", "efficientnet_b0"])
    ap.add_argument("--fusion_hidden_dim", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.architecture == "fusion" and args.image_size == 299:
        args.image_size = 224

    model_kwargs = {}
    if args.architecture == "fusion":
        clip_pretrained = None if args.clip_pretrained == "None" else args.clip_pretrained
        model_kwargs = dict(
            clip_model_name=args.clip_model_name, clip_pretrained=clip_pretrained,
            n_unfrozen_clip_blocks=args.n_unfrozen_clip_blocks, clip_proj_dim=args.clip_proj_dim,
            freq_feat_dim=args.freq_feat_dim, fusion_hidden_dim=args.fusion_hidden_dim,
            freq_backbone=args.freq_backbone,
        )

    print(f"Loading {args.architecture} model from {args.checkpoint}...")
    model = load_model(args.architecture, args.checkpoint, device, **model_kwargs)
    model.eval()

    df = pd.read_csv(args.manifest)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    print(f"Loaded {len(test_df)} test samples from {args.manifest}")

    robustness_results = {}

    for q in args.qualities:
        print(f"\n--- Evaluating Compression Quality Q = {q} ---")
        transform = build_compressed_transform(args.image_size, quality=q)
        
        labels, probs = [], []
        
        for idx in tqdm(range(0, len(test_df), args.batch_size), desc=f"JPEG Q={q}"):
            batch_df = test_df.iloc[idx : idx + args.batch_size]
            batch_imgs = []
            
            for _, row in batch_df.iterrows():
                img_path = row["path"] if "path" in row else row.get("filepath", "")
                img = cv2.imread(str(img_path))
                if img is None:
                    img = np.zeros((args.image_size, args.image_size, 3), dtype=np.uint8)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                t_img = transform(image=img)["image"]
                batch_imgs.append(t_img)
            
            t_batch = torch.stack(batch_imgs).to(device)
            with torch.no_grad():
                logits = model(t_batch)
                batch_probs = torch.sigmoid(logits).cpu().numpy()
            
            probs.extend(batch_probs)
            labels.extend(batch_df["label"].values)

        metrics = compute_metrics(labels, probs)
        robustness_results[f"Q_{q}"] = metrics
        print(f"Q={q} -> AUC: {metrics['auc']:.4f} | AP: {metrics['ap']:.4f} | EER: {metrics['eer']:.4f} | Acc: {metrics['acc']:.4f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(robustness_results, f, indent=2)

    print(f"\nCompression Robustness results written to {out_path}")


if __name__ == "__main__":
    main()
