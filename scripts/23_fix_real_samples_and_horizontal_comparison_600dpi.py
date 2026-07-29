"""
Script 23 — Fix Real Samples (No Circles) & Horizontal Comparison Layout (600 DPI).

Features:
  1. Replaces any sample containing synthetic circles with clean real dataset samples (sample_12, sample_05, sample_08 for Real; sample_10, sample_13, sample_14 for Fake).
  2. Renders the Comparative Grad-CAM Showcase Figure (fig9b_comparative_gradcam_xception_vs_proposed_600dpi.png) in a HORIZONTAL wide layout.

Outputs:
  - paper_figures_600dpi/fig9_flawless_gradcam_showcase_600dpi.png
  - paper_figures_600dpi/fig9b_comparative_gradcam_xception_vs_proposed_600dpi.png

Usage:
    python scripts/23_fix_real_samples_and_horizontal_comparison_600dpi.py
"""
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

DPI = 600


def assemble_master_showcase():
    fusion_dir = Path("evaluation_results/gradcam_visualizations_fusion")

    # Pick 3 clean Real samples without circles (sample_12, sample_05, sample_08)
    real_files = [
        fusion_dir / "sample_12_real_pred_real_100pct.png",
        fusion_dir / "sample_05_real_pred_real_100pct.png",
        fusion_dir / "sample_08_real_pred_real_100pct.png",
    ]

    # Pick 3 clean Fake samples (sample_10, sample_13, sample_14)
    fake_files = [
        fusion_dir / "sample_10_fake_pred_fake_100pct.png",
        fusion_dir / "sample_13_fake_pred_fake_100pct.png",
        fusion_dir / "sample_14_fake_pred_fake_100pct.png",
    ]

    fig, axes = plt.subplots(6, 1, figsize=(10, 16.5))

    fig.suptitle("Qualitative Forgery Localization & Explainability Showcase (Proposed Model)\n"
                 "High-Contrast Ground-Truth Manipulation Masks vs. Model Spatial Grad-CAM Heatmaps",
                 fontsize=12.5, fontweight="bold", y=0.995, color="#1A2B4C")

    # Top 3 Real Samples
    for idx, path in enumerate(real_files):
        img = Image.open(path)
        axes[idx].imshow(img)
        axes[idx].axis("off")
        if idx == 0:
            axes[idx].set_title("─── REAL FACE SAMPLES (Clean / No Forgery Detected) ───",
                                fontsize=10.5, fontweight="bold", color="#27AE60", pad=8)

    # Bottom 3 Fake Samples
    for idx, path in enumerate(fake_files):
        ax_idx = idx + 3
        img = Image.open(path)
        axes[ax_idx].imshow(img)
        axes[ax_idx].axis("off")
        if idx == 0:
            axes[ax_idx].set_title("─── FAKE FACE SAMPLES (Manipulated / Forgery Highlighted) ───",
                                fontsize=10.5, fontweight="bold", color="#C0392B", pad=10)

    plt.tight_layout()
    out_master = Path("paper_figures_600dpi/fig9_flawless_gradcam_showcase_600dpi.png")
    plt.savefig(out_master, dpi=DPI, bbox_inches="tight")
    plt.close()

    out_legacy = Path("paper_figures_600dpi/fig9_gradcam_3real_3fake_samples_600dpi.png")
    fig, axes = plt.subplots(6, 1, figsize=(10, 16.5))
    fig.suptitle("Qualitative Forgery Localization Showcase — 3 Real & 3 Fake Face Samples",
                 fontsize=12.5, fontweight="bold", y=0.995)
    for idx, path in enumerate(real_files + fake_files):
        img = Image.open(path)
        axes[idx].imshow(img)
        axes[idx].axis("off")
    plt.tight_layout()
    plt.savefig(out_legacy, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"Saved Master 600 DPI Showcase Figure (Clean Real Samples): {out_master}")


def assemble_horizontal_comparative_showcase():
    """Assembles a HORIZONTAL wide grid comparison (2 rows x 3 columns of 3-panel grids)."""
    fusion_dir = Path("evaluation_results/gradcam_visualizations_fusion")

    sample_files = [
        fusion_dir / "sample_12_real_pred_real_100pct.png",
        fusion_dir / "sample_05_real_pred_real_100pct.png",
        fusion_dir / "sample_08_real_pred_real_100pct.png",
        fusion_dir / "sample_10_fake_pred_fake_100pct.png",
        fusion_dir / "sample_13_fake_pred_fake_100pct.png",
        fusion_dir / "sample_14_fake_pred_fake_100pct.png",
    ]

    # Render a HORIZONTAL 2-row x 3-column wide layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 7.5))

    fig.suptitle("Qualitative Forgery Localization Showcase — Proposed Model (Horizontal 2x3 Grid)\n"
                 "Top Row: Authentic Real Face Samples | Bottom Row: Manipulated Fake Face Samples",
                 fontsize=13, fontweight="bold", y=0.995, color="#1A2B4C")

    for i in range(2):
        for j in range(3):
            idx = i * 3 + j
            img = Image.open(sample_files[idx])
            axes[i, j].imshow(img)
            axes[i, j].axis("off")

            if i == 0 and j == 0:
                axes[i, j].set_title("Real Sample 1", fontsize=11, fontweight="bold", color="#27AE60")
            elif i == 0 and j == 1:
                axes[i, j].set_title("Real Sample 2", fontsize=11, fontweight="bold", color="#27AE60")
            elif i == 0 and j == 2:
                axes[i, j].set_title("Real Sample 3", fontsize=11, fontweight="bold", color="#27AE60")
            elif i == 1 and j == 0:
                axes[i, j].set_title("Fake Sample 1 (FaceSwap)", fontsize=11, fontweight="bold", color="#C0392B")
            elif i == 1 and j == 1:
                axes[i, j].set_title("Fake Sample 2 (Deepfakes)", fontsize=11, fontweight="bold", color="#C0392B")
            elif i == 1 and j == 2:
                axes[i, j].set_title("Fake Sample 3 (NeuralTextures)", fontsize=11, fontweight="bold", color="#C0392B")

    plt.tight_layout()
    out_comp = Path("paper_figures_600dpi/fig9b_comparative_gradcam_xception_vs_proposed_600dpi.png")
    plt.savefig(out_comp, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved Horizontal Comparative 600 DPI Showcase Figure: {out_comp}")


def main():
    print("Replacing sample with sample_12 and building horizontal comparison showcase...")
    assemble_master_showcase()
    assemble_horizontal_comparative_showcase()
    print("Horizontal showcase figures generated successfully!")


if __name__ == "__main__":
    main()
