"""
Script 16 — Fix Table IV (Compression Degradation) Formatting at 600 DPI.

Ensures zero text overflow, generous column padding, and multi-line headers.
Output: paper_figures_600dpi/fig12_table4_compression_rendered_600dpi.png
"""
from pathlib import Path
import matplotlib.pyplot as plt

DPI = 600


def main():
    t4_data = [
        ["JPEG Quality\nFactor (Q)", "Compression Level\n& Scenario", "Proposed Model\nAUC (FF++)", "Proposed Model\nAUC (DFF)", "Xception AUC\n(FF++)"],
        ["Q = 100", "Clean / Uncompressed", "90.28%", "56.94%", "98.33%"],
        ["Q = 90", "Light Social Media", "90.27%", "56.92%", "98.33%"],
        ["Q = 80", "Standard Web Re-encode", "90.30%", "56.95%", "98.33%"],
        ["Q = 70", "Medium WhatsApp / Twitter", "90.29%", "56.93%", "98.32%"],
        ["Q = 60", "Heavy Compression", "90.26%", "56.91%", "98.33%"],
        ["Q = 50", "Severe Compression", "90.26%", "56.90%", "98.32%"],
    ]

    col_widths = [0.18, 0.28, 0.18, 0.18, 0.18]

    fig, ax = plt.subplots(figsize=(13.5, 5.5))
    ax.axis("off")
    ax.set_title("Table IV: Controlled Compression Degradation Benchmark across Both Datasets",
                 fontsize=13, fontweight="bold", pad=18, color="#1A2B4C")

    table = ax.table(
        cellText=t4_data[1:],
        colLabels=t4_data[0],
        cellLoc="center",
        loc="center",
        colWidths=col_widths
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.15, 2.2)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#BDC3C7")
        cell.set_linewidth(1.2)
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#2C3E50")
            cell.set_height(0.12)
        else:
            if row % 2 == 1:
                cell.set_facecolor("#F8F9FA")
            else:
                cell.set_facecolor("#FFFFFF")
            # Highlight Proposed Model columns
            if col in (2, 3):
                cell.set_facecolor("#E8F8F5")
                cell.set_text_props(weight="bold")

    plt.tight_layout()
    out = Path("paper_figures_600dpi/fig12_table4_compression_rendered_600dpi.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved Perfectly Formatted Table IV Image at 600 DPI: {out}")


if __name__ == "__main__":
    main()
