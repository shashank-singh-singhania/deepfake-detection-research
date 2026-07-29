"""
Script 12 — Generate Complete Publication Graphic Suite (Graphs & Rendered Table Images).

Generates 10 high-resolution 300 DPI figures in paper_figures/:
  1. fig1_in_dataset_benchmark.png      (FF++ C23 AUC & AP comparison)
  2. fig2_cross_domain_dff.png          (Zero-Shot Diffusion Model Generalization)
  3. fig3_compression_robustness.png    (JPEG Q=100 to Q=50 Stability Curve)
  4. fig4_per_method_auc_breakdown.png  (Per-method AUC across DF, F2F, FS, NT)
  5. fig5_pointing_game_mask_iou.png    (Localization Pointing Game & Mask IoU)
  6. fig6_ablation_study_gains.png      (Ablation gains: Fusion v3 vs Fusion v4)
  7. fig7_heatmap_intensity_stats.png   (Model Heatmap Activation Percentiles)
  8. fig8_table1_rendered.png           (Rendered Table Image: In-Dataset Benchmark)
  9. fig9_table2_rendered.png           (Rendered Table Image: Zero-Shot Cross-Domain)
 10. fig10_table3_rendered.png          (Rendered Table Image: Compression Robustness)

Usage:
    python scripts/12_generate_all_paper_graphics.py
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
    models = ["Xception\nBaseline", "Premier Fusion v4\n(EfficientNet-B0)", "Fusion v3\n(Simple CNN)", "TriConsistency\nNet", "SBI\nBaseline"]
    auc = [98.38, 91.07, 87.85, 80.66, 71.10]
    ap = [99.62, 97.59, 96.39, 93.97, 90.31]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    rects1 = ax.bar(x - width/2, auc, width, label="AUC (%)", color="#1f77b4", edgecolor="black")
    rects2 = ax.bar(x + width/2, ap, width, label="Average Precision (%)", color="#2ca02c", edgecolor="black")

    ax.set_ylabel("Percentage (%)", fontsize=12, fontweight="bold")
    ax.set_title("Figure 1: In-Dataset Performance Benchmark on FaceForensics++ (FF++) C23", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax.set_ylim(60, 103)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

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


def make_fig2_cross_domain():
    methods = ["InsightFace\n(Face Swap)", "SD Inpainting\n(Diffusion Inpaint)", "SD Text2Img\n(Diffusion Gen)", "Overall DFF\nCross-Domain"]
    xception_auc = [56.95, 52.16, 44.42, 51.18]
    fusion_auc = [62.42, 55.90, 52.50, 56.94]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    rects1 = ax.bar(x - width/2, xception_auc, width, label="Xception Baseline (FF++ Trained)", color="#d62728", edgecolor="black")
    rects2 = ax.bar(x + width/2, fusion_auc, width, label="Premier Fusion v4 (FF++ Trained)", color="#1f77b4", edgecolor="black")

    ax.set_ylabel("Zero-Shot Cross-Domain AUC (%)", fontsize=12, fontweight="bold")
    ax.set_title("Figure 2: Zero-Shot Cross-Domain Generalization on DeepFakeFace (Diffusion Models)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10, fontweight="bold")
    ax.set_ylim(35, 72)
    ax.axhline(50.0, color="gray", linestyle=":", linewidth=1.5, label="Random Guess Threshold (50%)")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

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


def make_fig3_compression():
    q_factors = [100, 90, 80, 70, 60, 50]
    fusion_auc = [90.28, 90.27, 90.30, 90.29, 90.26, 90.26]
    xception_auc = [98.33, 98.33, 98.33, 98.32, 98.33, 98.32]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(q_factors, xception_auc, "o-", color="#d62728", linewidth=2.5, markersize=7, label="Xception Baseline")
    ax.plot(q_factors, fusion_auc, "s-", color="#1f77b4", linewidth=2.5, markersize=7, label="Premier Fusion v4 (Gated)")

    ax.set_xlabel("JPEG Quality Factor (Q)", fontsize=12, fontweight="bold")
    ax.set_ylabel("AUC (%)", fontsize=12, fontweight="bold")
    ax.set_title("Figure 3: Controlled JPEG Compression Robustness Evaluation (Q=100 to Q=50)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(105, 45)
    ax.set_ylim(85, 100)
    ax.legend(fontsize=11, loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out = Path("paper_figures/fig3_compression_robustness.png")
    plt.savefig(out, dpi=300)
    plt.close()


def make_fig4_per_method():
    """Figure 4: Per-Method Forgery AUC Breakdown across DF, F2F, FS, NT."""
    methods = ["Deepfakes (DF)", "Face2Face (F2F)", "FaceSwap (FS)", "NeuralTextures (NT)"]
    xception = [99.14, 98.88, 98.45, 97.03]
    fusion_v4 = [95.06, 91.88, 91.88, 85.47]
    fusion_v3 = [93.62, 87.41, 89.23, 81.13]
    tricon = [84.37, 81.13, 79.93, 77.22]

    x = np.arange(len(methods))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 1.5*width, xception, width, label="Xception Baseline", color="#d62728", edgecolor="black")
    ax.bar(x - 0.5*width, fusion_v4, width, label="Premier Fusion v4", color="#1f77b4", edgecolor="black")
    ax.bar(x + 0.5*width, fusion_v3, width, label="Fusion v3 (Simple CNN)", color="#ff7f0e", edgecolor="black")
    ax.bar(x + 1.5*width, tricon, width, label="TriConsistencyNet", color="#9467bd", edgecolor="black")

    ax.set_ylabel("AUC (%)", fontsize=12, fontweight="bold")
    ax.set_title("Figure 4: Per-Method Forgery Detection AUC Breakdown", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10, fontweight="bold")
    ax.set_ylim(70, 102)
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out = Path("paper_figures/fig4_per_method_auc_breakdown.png")
    plt.savefig(out, dpi=300)
    plt.close()


def make_fig5_pointing_iou():
    """Figure 5: Spatial Localization Performance (Pointing Game Accuracy & Mask IoU)."""
    metrics = ["Pointing Game Accuracy (%)", "Adaptive Mask IoU (%)"]
    fusion_v3 = [75.55, 41.76]
    fusion_v4 = [88.84, 57.84]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    rects1 = ax.bar(x - width/2, fusion_v3, width, label="Fusion v3 (Simple CNN)", color="#ff7f0e", edgecolor="black")
    rects2 = ax.bar(x + width/2, fusion_v4, width, label="Fusion v4 (EfficientNet-B0)", color="#1f77b4", edgecolor="black")

    ax.set_ylabel("Percentage (%)", fontsize=12, fontweight="bold")
    ax.set_title("Figure 5: Spatial Localization Explainability Metrics", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11, fontweight="bold")
    ax.set_ylim(30, 100)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}%", xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}%\n(+{height-fusion_v3[rects2.index(rect)]:.2f}%)", xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#1f77b4")

    plt.tight_layout()
    out = Path("paper_figures/fig5_pointing_game_mask_iou.png")
    plt.savefig(out, dpi=300)
    plt.close()


def make_fig6_ablation_gains():
    """Figure 6: Ablation Gains (Fusion v3 vs Fusion v4)."""
    categories = ["Classification\nAUC (+3.22%)", "Pointing Game\nAcc (+13.29%)", "Adaptive Mask\nIoU (+16.08%)"]
    gains = [3.22, 13.29, 16.08]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(categories, gains, color=["#2ca02c", "#1f77b4", "#9467bd"], edgecolor="black", width=0.5)

    ax.set_ylabel("Net Performance Increase (%)", fontsize=12, fontweight="bold")
    ax.set_title("Figure 6: Ablation Gains from SE-Attention EfficientNet-B0 Backbone", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, 20)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"+{height:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    out = Path("paper_figures/fig6_ablation_study_gains.png")
    plt.savefig(out, dpi=300)
    plt.close()


def render_table_image(table_data, col_widths, title, filename):
    """Renders a clean, publication-grade table image using matplotlib."""
    fig, ax = plt.subplots(figsize=(10, len(table_data) * 0.6 + 1.2))
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15, color="#1A2B4C")

    table = ax.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc="center",
        loc="center",
        colWidths=col_widths
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style Header
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#BDC3C7")
        cell.set_linewidth(1.0)
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#2C3E50")
        else:
            if row % 2 == 1:
                cell.set_facecolor("#F8F9FA")
            else:
                cell.set_facecolor("#FFFFFF")
            # Highlight Premier Fusion v4
            if "Premier Fusion v4" in table_data[row][0]:
                cell.set_facecolor("#E8F8F5")
                cell.set_text_props(weight="bold")

    plt.tight_layout()
    out = Path(f"paper_figures/{filename}")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def make_rendered_tables():
    """Renders Table I, II, III as high-res images."""
    # Table 1: In-Dataset Benchmark
    t1_data = [
        ["Model Architecture", "Overall AUC", "AP", "EER", "Acc", "Pointing Acc", "Mask IoU"],
        ["Xception Baseline", "98.38%", "99.62%", "5.20%", "95.18%", "N/A", "N/A"],
        ["Premier Fusion v4", "91.07%", "97.59%", "17.15%", "82.28%", "88.84%", "57.84%"],
        ["Fusion v3 (Simple CNN)", "87.85%", "96.39%", "19.58%", "82.42%", "75.55%", "41.76%"],
        ["TriConsistencyNet", "80.66%", "93.97%", "27.22%", "80.77%", "N/A", "N/A"],
        ["SBI Baseline", "71.10%", "90.31%", "34.96%", "25.95%", "N/A", "N/A"],
    ]
    render_table_image(t1_data, [0.26, 0.12, 0.12, 0.12, 0.12, 0.13, 0.13], "Table I: In-Dataset Performance Benchmark (FF++ C23)", "fig8_table1_rendered.png")

    # Table 2: Zero-Shot Cross-Domain
    t2_data = [
        ["Model Architecture", "Overall Cross AUC", "InsightFace", "SD Inpainting", "SD Text2Img"],
        ["Xception Baseline", "51.18%", "56.95%", "52.16%", "44.42%"],
        ["Premier Fusion v4", "56.94%", "62.42%", "55.90%", "52.50%"],
        ["Net Fusion Advantage", "+5.76%", "+5.47%", "+3.74%", "+8.08%"],
    ]
    render_table_image(t2_data, [0.28, 0.18, 0.18, 0.18, 0.18], "Table II: Zero-Shot Cross-Domain Performance (DeepFakeFace)", "fig9_table2_rendered.png")

    # Table 3: Compression Robustness
    t3_data = [
        ["JPEG Quality Factor (Q)", "Compression Level", "Premier Fusion v4 AUC", "Fusion v4 AP", "Xception AUC"],
        ["Q = 100", "Clean / Uncompressed", "90.28%", "97.32%", "98.33%"],
        ["Q = 90", "Light Social Media", "90.27%", "97.32%", "98.33%"],
        ["Q = 80", "Standard Web Re-encode", "90.30%", "97.32%", "98.33%"],
        ["Q = 70", "Medium WhatsApp/Twitter", "90.29%", "97.33%", "98.32%"],
        ["Q = 60", "Heavy Compression", "90.26%", "97.32%", "98.33%"],
        ["Q = 50", "Severe Compression", "90.26%", "97.32%", "98.32%"],
    ]
    render_table_image(t3_data, [0.22, 0.25, 0.20, 0.18, 0.15], "Table III: Controlled Compression Degradation Benchmark (JPEG Q=100..50)", "fig10_table3_rendered.png")


def main():
    print("Generating complete publication graphic suite...")
    make_fig1_in_dataset()
    make_fig2_cross_domain()
    make_fig3_compression()
    make_fig4_per_method()
    make_fig5_pointing_iou()
    make_fig6_ablation_gains()
    make_rendered_tables()
    print("All 9 publication figures and rendered table images saved in paper_figures/!")


if __name__ == "__main__":
    main()
