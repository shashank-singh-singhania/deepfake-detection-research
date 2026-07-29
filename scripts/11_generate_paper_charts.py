"""
Script 11 — Generate High-Resolution Publication Charts & Figures for Research Paper.

Generates:
  1. paper_figures/fig1_in_dataset_benchmark.png (FF++ C23 AUC, AP, EER comparison)
  2. paper_figures/fig2_cross_domain_dff.png (Zero-Shot Diffusion Model Generalization)
  3. paper_figures/fig3_compression_robustness.png (JPEG Q=100 to Q=50 Stability Curve)
  4. paper_figures/fig4_ablation_study.png (Backbone & Stream Ablation Gains)

Usage:
    python scripts/11_generate_paper_charts.py
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 1.0


def make_fig1_in_dataset():
    """Figure 1: In-Dataset Performance Comparison on FaceForensics++ C23."""
    models = ["Xception\nBaseline", "Premier Fusion v4\n(EfficientNet-B0)", "Fusion v3\n(Simple CNN)", "TriConsistency\nNet", "SBI\nBaseline"]
    auc = [98.38, 91.07, 87.85, 80.66, 71.10]
    ap = [99.62, 97.59, 96.39, 93.97, 90.31]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    rects1 = ax.bar(x - width/2, auc, width, label="AUC (%)", color="#1f77b4", edgecolor="black")
    rects2 = ax.bar(x + width/2, ap, width, label="Average Precision (%)", color="#2ca02c", edgecolor="black")

    ax.set_ylabel("Percentage (%)", fontsize=12, fontweight="bold")
    ax.set_title("In-Dataset Performance Benchmark on FaceForensics++ (FF++) C23", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax.set_ylim(60, 103)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Value labels
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out = Path("paper_figures/fig1_in_dataset_benchmark.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


def make_fig2_cross_domain():
    """Figure 2: Zero-Shot Cross-Domain Generalization on DeepFakeFace (Diffusion Models)."""
    methods = ["InsightFace\n(Face Swap)", "SD Inpainting\n(Diffusion Inpaint)", "SD Text2Img\n(Diffusion Gen)", "Overall DFF\nCross-Domain"]
    xception_auc = [56.95, 52.16, 44.42, 51.18]
    fusion_auc = [62.42, 55.90, 52.50, 56.94]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    rects1 = ax.bar(x - width/2, xception_auc, width, label="Xception Baseline (FF++ Trained)", color="#d62728", edgecolor="black")
    rects2 = ax.bar(x + width/2, fusion_auc, width, label="Premier Fusion v4 (FF++ Trained)", color="#1f77b4", edgecolor="black")

    ax.set_ylabel("Zero-Shot Cross-Domain AUC (%)", fontsize=12, fontweight="bold")
    ax.set_title("Zero-Shot Cross-Domain Generalization on DeepFakeFace (Diffusion Models)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10, fontweight="bold")
    ax.set_ylim(35, 72)
    ax.axhline(50.0, color="gray", linestyle=":", linewidth=1.5, label="Random Guess Threshold (50%)")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Value labels
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1f77b4")

    plt.tight_layout()
    out = Path("paper_figures/fig2_cross_domain_dff.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


def make_fig3_compression():
    """Figure 3: Image & Video Compression Degradation Benchmark (JPEG Q=100 to Q=50)."""
    q_factors = [100, 90, 80, 70, 60, 50]
    fusion_auc = [90.28, 90.27, 90.30, 90.29, 90.26, 90.26]
    xception_auc = [98.33, 98.33, 98.33, 98.32, 98.33, 98.32]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(q_factors, xception_auc, "o-", color="#d62728", linewidth=2.5, markersize=7, label="Xception Baseline")
    ax.plot(q_factors, fusion_auc, "s-", color="#1f77b4", linewidth=2.5, markersize=7, label="Premier Fusion v4 (Gated)")

    ax.set_xlabel("JPEG Quality Factor (Q)", fontsize=12, fontweight="bold")
    ax.set_ylabel("AUC (%)", fontsize=12, fontweight="bold")
    ax.set_title("Controlled JPEG Compression Robustness Evaluation (Q=100 to Q=50)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(105, 45) # Inverted axis so 100 is left, 50 is right
    ax.set_ylim(85, 100)
    ax.legend(fontsize=11, loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out = Path("paper_figures/fig3_compression_robustness.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


def main():
    print("Generating publication charts...")
    make_fig1_in_dataset()
    make_fig2_cross_domain()
    make_fig3_compression()
    print("All paper figures generated in paper_figures/!")


if __name__ == "__main__":
    main()
