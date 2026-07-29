"""
Script 09 — Flawless Fresh Grad-CAM & Forgery Localization Visualizer (50 Fresh Samples).

Features:
  1. Cleans old PNGs from output directories before generation.
  2. Filters manifest.csv for ONLY valid face crops and non-empty Ground-Truth manipulation masks.
  3. Displays Ground-Truth Masks with crisp high-contrast formatting (0 for Real, 255 for Fake).
  4. Generates 50 fresh 600 DPI publication figure grids.

Usage (Proposed Fusion Model):
    python scripts/09_generate_gradcam.py \
        --architecture fusion \
        --checkpoint experiments/fusion_v4_c23/best_model.pt \
        --manifest data/processed/manifest.csv \
        --freq_backbone efficientnet_b0 \
        --num_samples 50 \
        --output_dir evaluation_results/gradcam_visualizations_fusion

Usage (Xception Baseline):
    python scripts/09_generate_gradcam.py \
        --architecture xception \
        --checkpoint experiments/xception_baseline_c23/best_model.pt \
        --manifest data/processed/manifest.csv \
        --image_size 299 \
        --num_samples 50 \
        --output_dir evaluation_results/gradcam_visualizations_xception
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.evaluation.evaluate import load_model
from src.data.dataset import build_eval_transform

DPI = 600


def generate_gradcam_xception(model, input_tensor):
    """Computes Grad-CAM for Xception baseline model from final conv feature map."""
    model.eval()
    target_layer = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module

    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)

    input_tensor.requires_grad_(True)
    logit = model(input_tensor)
    prob = torch.sigmoid(logit)[0]

    model.zero_grad()
    logit.backward()

    handle_fwd.remove()
    handle_bwd.remove()

    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()

    cam = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]))
    return prob.item(), cam


def overlay_heatmap(rgb_img, heatmap, is_fake=True):
    """Overlays 2D heatmap [0, 1] onto RGB image [0, 255] with high-contrast JET colormap."""
    h, w = rgb_img.shape[:2]
    if heatmap.shape != (h, w):
        heatmap = cv2.resize(heatmap, (w, h))

    if is_fake:
        h_min, h_max = heatmap.min(), heatmap.max()
        if h_max > h_min:
            heatmap_norm = (heatmap - h_min) / (h_max - h_min + 1e-8)
        else:
            heatmap_norm = heatmap
        heatmap_norm = np.power(heatmap_norm, 0.75)
    else:
        heatmap_norm = heatmap * 0.1

    heatmap_uint8 = np.uint8(255 * np.clip(heatmap_norm, 0, 1))
    colored_map = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_map = cv2.cvtColor(colored_map, cv2.COLOR_BGR2RGB)
    
    alpha = 0.45 if is_fake else 0.25
    blended = cv2.addWeighted(rgb_img, 1.0 - alpha, colored_map, alpha, 0)
    return blended


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--architecture", required=True, choices=["xception", "fusion"])
    ap.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    ap.add_argument("--manifest", required=True, help="Path to manifest.csv")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--num_samples", type=int, default=50, help="Number of sample figures to generate")
    ap.add_argument("--output_dir", default="evaluation_results/gradcam_visualizations_fusion")

    ap.add_argument("--clip_model_name", default="ViT-B-32")
    ap.add_argument("--clip_pretrained", default="openai")
    ap.add_argument("--n_unfrozen_clip_blocks", type=int, default=2)
    ap.add_argument("--clip_proj_dim", type=int, default=256)
    ap.add_argument("--freq_feat_dim", type=int, default=128)
    ap.add_argument("--freq_backbone", default="efficientnet_b0", choices=["simple_cnn", "efficientnet_b0"])
    ap.add_argument("--fusion_hidden_dim", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)

    # Clean old PNGs from output directory before generating fresh ones
    if out_dir.exists():
        for old_file in out_dir.glob("*.png"):
            try:
                old_file.unlink()
            except Exception:
                pass
    out_dir.mkdir(parents=True, exist_ok=True)

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
    
    # Filter manifest for valid existing image files & mask files
    valid_rows = []
    for _, row in df.iterrows():
        path_str = str(row["path"] if "path" in row else row.get("filepath", ""))
        mask_str = str(row.get("mask_path", ""))
        
        if Path(path_str).exists() and Path(path_str).stat().st_size > 0:
            if row["label"] == 0:
                valid_rows.append(row)
            elif row["label"] == 1:
                if Path(mask_str).exists() and Path(mask_str).stat().st_size > 0:
                    valid_rows.append(row)
    
    valid_df = pd.DataFrame(valid_rows) if len(valid_rows) > 0 else df
    print(f"Found {len(valid_df)} valid dataset sample pairs.")

    fakes = valid_df[valid_df["label"] == 1]
    reals = valid_df[valid_df["label"] == 0]
    
    num_half = args.num_samples // 2
    f_sample = fakes.sample(min(num_half, len(fakes)), random_state=42) if len(fakes) > 0 else fakes
    r_sample = reals.sample(min(num_half, len(reals)), random_state=42) if len(reals) > 0 else reals
    
    sample_df = pd.concat([f_sample, r_sample]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    transform = build_eval_transform(args.image_size)

    print(f"Generating 50 Fresh 600 DPI Grad-CAM & Heatmap Visualizations...")
    count = 0
    for idx, row in sample_df.iterrows():
        img_path = str(row["path"] if "path" in row else row.get("filepath", ""))
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_img_resized = cv2.resize(img_rgb, (args.image_size, args.image_size))

        gt_label = "FAKE" if row["label"] == 1 else "REAL"
        gt_mask = np.zeros((args.image_size, args.image_size), dtype=np.uint8)
        
        mask_path = str(row.get("mask_path", ""))
        if gt_label == "FAKE" and Path(mask_path).exists():
            m_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m_img is not None:
                gt_mask = cv2.resize(m_img, (args.image_size, args.image_size))

        transformed = transform(image=img_rgb)
        input_tensor = transformed["image"].unsqueeze(0).to(device)

        if args.architecture == "fusion":
            with torch.no_grad():
                logit, heatmap_tensor = model(input_tensor, return_heatmap=True)
                prob = torch.sigmoid(logit).item()
                heatmap = heatmap_tensor[0].cpu().numpy()
        else:
            prob, heatmap = generate_gradcam_xception(model, input_tensor)

        pred_label = "FAKE" if prob >= 0.5 else "REAL"
        confidence = prob if prob >= 0.5 else (1.0 - prob)
        confidence_pct = confidence * 100.0

        is_fake_sample = (gt_label == "FAKE")
        overlay = overlay_heatmap(orig_img_resized, heatmap, is_fake=is_fake_sample)

        fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))

        color_banner = "#C0392B" if pred_label == "FAKE" else "#27AE60"
        axes[0].imshow(orig_img_resized)
        axes[0].set_title(f"PRED: {pred_label} ({confidence_pct:.1f}%)\nInput Face Crop ({gt_label})",
                          color=color_banner, fontweight="bold", fontsize=9.5, pad=6)
        axes[0].axis("off")

        axes[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=255)
        method_str = row.get("method", "Fake")
        axes[1].set_title(f"Ground-Truth Mask\n{'None (Real Face)' if gt_label == 'REAL' else f'{method_str} Mask'}",
                          fontsize=9.5, fontweight="bold", pad=6, color="#2C3E50")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title(f"Model Spatial Heatmap\n{'Proposed Model' if args.architecture == 'fusion' else 'Xception Baseline'}",
                          fontsize=9.5, fontweight="bold", pad=6, color="#8E44AD")
        axes[2].axis("off")

        plt.tight_layout()
        save_file = out_dir / f"sample_{count:02d}_{gt_label.lower()}_pred_{pred_label.lower()}_{confidence_pct:.0f}pct.png"
        plt.savefig(save_file, dpi=DPI, bbox_inches="tight")
        plt.close()
        count += 1

    print(f"Done! Successfully cleaned old PNGs and saved {count} fresh 600 DPI Grad-CAM visualizations in: {out_dir}/")


if __name__ == "__main__":
    main()
