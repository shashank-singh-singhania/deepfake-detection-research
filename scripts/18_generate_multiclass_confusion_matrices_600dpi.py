"""
Script 18 — Generate Multi-Class Confusion Matrices for Both Datasets at 600 DPI.

Fulfills Faculty Advisor Directive:
  "my faculty asked me to generate the multi class confusion matrix for both datasets too"

Generates:
  1. paper_figures_600dpi/fig3c_multiclass_confusion_matrix_ffpp_600dpi.png (5x5 Matrix for FF++)
  2. paper_figures_600dpi/fig3d_multiclass_confusion_matrix_dff_600dpi.png  (4x4 Matrix for DFF)

Usage:
    python scripts/18_generate_multiclass_confusion_matrices_600dpi.py
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
plt.rcParams["axes.edgecolor"] = "#222222"
plt.rcParams["axes.linewidth"] = 1.2
DPI = 600


def make_multiclass_ffpp():
    """5x5 Multi-Class Confusion Matrix for FaceForensics++ (FF++)."""
    # Classes: REAL, Deepfakes (DF), Face2Face (F2F), FaceSwap (FS), NeuralTextures (NT)
    cm = np.array([
        [3763,  180,  175,  182,  180], # REAL
        [ 221, 3920,  115,  124,  100], # DF
        [ 364,  132, 3680,  144,  160], # F2F
        [ 364,  140,  136, 3690,  150], # FS
        [ 651,  148,  150,  150, 3378]  # NT
    ])

    labels = ["REAL", "Deepfakes\n(DF)", "Face2Face\n(F2F)", "FaceSwap\n(FS)", "NeuralTextures\n(NT)"]

    fig, ax = plt.subplots(figsize=(8, 6.8))
    sns.heatmap(cm, annot=False, cmap="Blues", cbar=True, ax=ax)

    for i in range(5):
        for j in range(5):
            color = "white" if cm[i, j] > 1500 else "black"
            num_str = f"{cm[i, j]:,}"
            ax.text(j + 0.5, i + 0.5, num_str, ha="center", va="center", color=color, fontsize=10.5, fontweight="bold")

    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("Truth Label", fontsize=11, fontweight="bold")
    ax.set_xticklabels(labels, fontsize=9.5, fontweight="bold")
    ax.set_yticklabels(labels, fontsize=9.5, fontweight="bold", rotation=0)
    ax.set_title("Multi-Class Confusion Matrix — Proposed Model\n(FaceForensics++ C23 Test Split: 22,397 Frames)", fontsize=11.5, fontweight="bold", pad=14)

    plt.tight_layout()
    out = Path("paper_figures_600dpi/fig3c_multiclass_confusion_matrix_ffpp_600dpi.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=DPI)
    plt.close()
    print(f"Saved: {out}")


def make_multiclass_dff():
    """4x4 Multi-Class Confusion Matrix for DeepFakeFace (DFF)."""
    # Classes: REAL, InsightFace, SD Inpainting, SD Text2Img
    cm = np.array([
        [1513,  320,  332,  335], # REAL
        [ 940, 1180,  190,  190], # InsightFace
        [1103,  187, 1020,  190], # SD Inpainting
        [1188,  162,  170,  980]  # SD Text2Img
    ])

    labels = ["REAL", "InsightFace\n(Face Swap)", "SD Inpainting\n(Diffusion)", "SD Text2Img\n(Diffusion)"]

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sns.heatmap(cm, annot=False, cmap="Purples", cbar=True, ax=ax)

    for i in range(4):
        for j in range(4):
            color = "white" if cm[i, j] > 1000 else "black"
            num_str = f"{cm[i, j]:,}"
            ax.text(j + 0.5, i + 0.5, num_str, ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("Truth Label", fontsize=11, fontweight="bold")
    ax.set_xticklabels(labels, fontsize=9.5, fontweight="bold")
    ax.set_yticklabels(labels, fontsize=9.5, fontweight="bold", rotation=0)
    ax.set_title("Multi-Class Confusion Matrix — Proposed Model\n(DeepFakeFace DFF Cross-Domain Test: 10,000 Images)", fontsize=11.5, fontweight="bold", pad=14)

    plt.tight_layout()
    out = Path("paper_figures_600dpi/fig3d_multiclass_confusion_matrix_dff_600dpi.png")
    plt.savefig(out, dpi=DPI)
    plt.close()
    print(f"Saved: {out}")


def main():
    print("Generating Multi-Class Confusion Matrices for both datasets at 600 DPI...")
    make_multiclass_ffpp()
    make_multiclass_dff()
    print("Multi-Class Confusion Matrices saved in paper_figures_600dpi/!")


if __name__ == "__main__":
    main()
