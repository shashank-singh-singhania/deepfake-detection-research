"""
Script 14 — Generate Composite 600 DPI Grad-CAM & Heatmap Figure (3 Real & 3 Fake Samples).

Fulfills Faculty Advisor Directive:
  "GradCam samples to showcase properly, my faculty advised 3 samples fake/real"

Combines 3 Real Face Samples and 3 Fake Face Samples into a publication-ready 600 DPI figure:
  paper_figures_600dpi/fig9_gradcam_3real_3fake_samples_600dpi.png

Usage:
    python scripts/14_generate_gradcam_composite_600dpi.py
"""
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

DPI = 600


def main():
    gradcam_dir = Path("evaluation_results/gradcam_visualizations_fusion")

    real_files = [
        gradcam_dir / "sample_00_real_pred_real_100pct.png",
        gradcam_dir / "sample_01_real_pred_real_100pct.png",
        gradcam_dir / "sample_03_real_pred_real_100pct.png",
    ]

    fake_files = [
        gradcam_dir / "sample_02_fake_pred_fake_100pct.png",
        gradcam_dir / "sample_04_fake_pred_fake_100pct.png",
        gradcam_dir / "sample_06_fake_pred_fake_100pct.png",
    ]

    # Load images
    real_imgs = [Image.open(f) for f in real_files]
    fake_imgs = [Image.open(f) for f in fake_files]

    fig, axes = plt.subplots(6, 1, figsize=(10, 15))

    # Real Samples Header Title
    fig.suptitle("Qualitative Forgery Localization & Explainability Visualizations (Proposed Model)\n"
                 "3 Real Face Samples (Top) vs. 3 Fake Face Samples (Bottom)",
                 fontsize=12, fontweight="bold", y=0.995, color="#1A2B4C")

    # Plot Real Samples (Rows 0, 1, 2)
    for idx, img in enumerate(real_imgs):
        axes[idx].imshow(img)
        axes[idx].axis("off")
        if idx == 0:
            axes[idx].set_title("─── REAL FACE SAMPLES (Clean / No Forgery Detected) ───",
                                fontsize=10, fontweight="bold", color="#27AE60", pad=6)

    # Plot Fake Samples (Rows 3, 4, 5)
    for idx, img in enumerate(fake_imgs):
        ax_idx = idx + 3
        axes[ax_idx].imshow(img)
        axes[ax_idx].axis("off")
        if idx == 0:
            axes[ax_idx].set_title("─── FAKE FACE SAMPLES (Manipulated / Forgery Highlighted) ───",
                                fontsize=10, fontweight="bold", color="#C0392B", pad=8)

    plt.tight_layout()
    out = Path("paper_figures_600dpi/fig9_gradcam_3real_3fake_samples_600dpi.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved 600 DPI Grad-CAM Composite Figure: {out}")


if __name__ == "__main__":
    main()
