"""
Script 09 — Grad-CAM & Heatmap Explainability Visualizer with Prediction Confidence Rate.

Generates high-resolution side-by-side visualization grids showing:
  - Column 1: Original Input Face Crop + Model Prediction & Confidence Rate (%)
  - Column 2: Ground-Truth Manipulation Mask (for fake images)
  - Column 3: Model Spatial Heatmap / Grad-CAM Overlay

Usage (Fusion Model v4):
    python scripts/09_generate_gradcam.py \
        --architecture fusion \
        --checkpoint experiments/fusion_v4_c23/best_model.pt \
        --manifest data/processed/manifest.csv \
        --freq_backbone efficientnet_b0 \
        --num_samples 12 \
        --output_dir evaluation_results/gradcam_visualizations

Usage (Xception Baseline):
    python scripts/09_generate_gradcam.py \
        --architecture xception \
        --checkpoint experiments/xception_baseline_c23/best_model.pt \
        --manifest data/processed/manifest.csv \
        --image_size 299 \
        --num_samples 12 \
        --output_dir evaluation_results/gradcam_visualizations_xception
"""
import argparse
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


def generate_gradcam_xception(model, input_tensor):
    """Computes Grad-CAM for Xception baseline model from final conv feature map."""
    model.eval()
    target_layer = None
    # Find last Conv2d layer in xception model
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


def overlay_heatmap(rgb_img, heatmap, alpha=0.5):
    """Overlays 2D heatmap [0, 1] onto RGB image [0, 255]."""
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored_map = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_map = cv2.cvtColor(colored_map, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(rgb_img, 1.0 - alpha, colored_map, alpha, 0)
    return blended


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--architecture", required=True, choices=["xception", "fusion"])
    ap.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    ap.add_argument("--manifest", required=True, help="Path to manifest.csv")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--num_samples", type=int, default=12, help="Number of sample figures to generate")
    ap.add_argument("--output_dir", default="evaluation_results/gradcam_visualizations")

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
    out_dir = Path(args.output_dir)
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
    # Pick balanced samples (half real, half fake)
    fakes = df[df["label"] == 1].sample(min(args.num_samples // 2, len(df[df["label"] == 1])), random_state=42)
    reals = df[df["label"] == 0].sample(min(args.num_samples // 2, len(df[df["label"] == 0])), random_state=42)
    sample_df = pd.concat([fakes, reals]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    transform = build_eval_transform(args.image_size)

    print(f"Generating Grad-CAM & Heatmap Visualizations for {len(sample_df)} samples...")
    for idx, row in sample_df.iterrows():
        img_path = row["path"] if "path" in row else row.get("filepath", "")
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_img_resized = cv2.resize(img_rgb, (args.image_size, args.image_size))

        # Prepare mask
        gt_mask = np.zeros((args.image_size, args.image_size), dtype=np.float32)
        mask_path = row.get("mask_path", "")
        if isinstance(mask_path, str) and mask_path and Path(mask_path).exists():
            m_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m_img is not None:
                gt_mask = cv2.resize(m_img, (args.image_size, args.image_size)).astype(np.float32) / 255.0

        # Transform input tensor
        transformed = transform(image=img_rgb)
        input_tensor = transformed["image"].unsqueeze(0).to(device)

        # Compute prediction logit, prob, and heatmap
        if args.architecture == "fusion":
            with torch.no_grad():
                logit, heatmap_tensor = model(input_tensor, return_heatmap=True)
                prob = torch.sigmoid(logit).item()
                heatmap = heatmap_tensor[0].cpu().numpy()
        else:
            prob, heatmap = generate_gradcam_xception(model, input_tensor)

        # Prediction details
        gt_label = "FAKE" if row["label"] == 1 else "REAL"
        pred_label = "FAKE" if prob >= 0.5 else "REAL"
        confidence = prob if prob >= 0.5 else (1.0 - prob)
        confidence_pct = confidence * 100.0

        overlay = overlay_heatmap(orig_img_resized, heatmap)

        # Plot 1x3 Figure Grid
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Col 1: Original Image + Prediction & Confidence
        axes[0].imshow(orig_img_resized)
        color = "green" if pred_label == gt_label else "red"
        axes[0].set_title(f"Input ({gt_label})\nPred: {pred_label} ({confidence_pct:.1f}%)", color=color, fontweight="bold")
        axes[0].axis("off")

        # Col 2: Ground-Truth Mask
        axes[1].imshow(gt_mask, cmap="gray")
        axes[1].set_title("Ground-Truth Mask" if gt_label == "FAKE" else "GT Mask (Real: 0.0)")
        axes[1].axis("off")

        # Col 3: Heatmap Overlay
        axes[2].imshow(overlay)
        axes[2].set_title(f"Model Heatmap / Grad-CAM\n({args.architecture.upper()})")
        axes[2].axis("off")

        plt.tight_layout()
        save_file = out_dir / f"sample_{idx:02d}_{gt_label.lower()}_pred_{pred_label.lower()}_{confidence_pct:.0f}pct.png"
        plt.savefig(save_file, dpi=200, bbox_inches="tight")
        plt.close()

    print(f"All {len(sample_df)} Grad-CAM & Heatmap visualizations saved in: {out_dir}/")


if __name__ == "__main__":
    main()
