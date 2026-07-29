"""
Script 22 — Swap sample_01 for sample_03 and assemble 600 DPI showcase figures.

Replaces sample_01 with sample_03 (Real face sample) in the master showcase figures.

Usage:
    python scripts/22_swap_sample01_and_assemble_600dpi.py
"""
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

DPI = 600


def assemble_master_showcase():
    fusion_dir = Path("evaluation_results/gradcam_visualizations_fusion")

    # Swapped sample_01 -> sample_03 for Real faces
    real_files = [
        fusion_dir / "sample_03_real_pred_real_100pct.png",
        fusion_dir / "sample_05_real_pred_real_100pct.png",
        fusion_dir / "sample_08_real_pred_real_100pct.png",
    ]

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

    # Legacy copy
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

    print(f"Saved Master 600 DPI Showcase Figure (swapped sample_01 -> sample_03): {out_master}")


def assemble_comparative_showcase():
    fusion_dir = Path("evaluation_results/gradcam_visualizations_fusion")

    sample_ids = ["03_real", "05_real", "08_real", "10_fake", "13_fake", "14_fake"]

    fig, axes = plt.subplots(6, 1, figsize=(11, 16.5))
    fig.suptitle("Qualitative Forgery Explainability Comparison: Xception Baseline vs. Proposed Model\n"
                 "Left Grid: Proposed Model Spatial Heatmap | Right Grid: Xception Baseline Grad-CAM",
                 fontsize=12, fontweight="bold", y=0.995, color="#1A2B4C")

    for idx, sid in enumerate(sample_ids):
        f_match = list(fusion_dir.glob(f"sample_{sid}_*.png"))
        if f_match:
            img = Image.open(f_match[0])
            axes[idx].imshow(img)
            axes[idx].axis("off")
            if idx == 0:
                axes[idx].set_title("─── REAL SAMPLES (Clean Baseline) ───", fontsize=10, fontweight="bold", color="#27AE60", pad=6)
            elif idx == 3:
                axes[idx].set_title("─── FAKE SAMPLES (Precise Forgery Localization) ───", fontsize=10, fontweight="bold", color="#C0392B", pad=8)

    plt.tight_layout()
    out_comp = Path("paper_figures_600dpi/fig9b_comparative_gradcam_xception_vs_proposed_600dpi.png")
    plt.savefig(out_comp, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved Comparative 600 DPI Figure: {out_comp}")


def main():
    print("Swapping sample_01 -> sample_03 and assembling 600 DPI showcase figures...")
    assemble_master_showcase()
    assemble_comparative_showcase()
    print("Updated 600 DPI showcase figures saved!")


if __name__ == "__main__":
    main()
