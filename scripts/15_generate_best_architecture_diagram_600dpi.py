"""
Script 15 — Generate Perfectly Aligned 600 DPI System Architecture Diagram.

Fixes & Improvements:
  1. Perfect centering & alignment for all embedded image sub-axes (Input, SRM, Heatmap).
  2. Uses sample_02_fake_pred_fake_100pct.png for a vivid, high-contrast forgery heatmap.
  3. Clean container proportions and crisp typography at 600 DPI.
  4. Output: paper_figures_600dpi/fig1_best_system_architecture_diagram_600dpi.png

Usage:
    python scripts/15_generate_best_architecture_diagram_600dpi.py
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2

DPI = 600


def get_sample_crops():
    """Extracts actual Input Face Crop, SRM Filter Residual, and Heatmap from sample_02."""
    sample_path = Path("evaluation_results/gradcam_visualizations_fusion/sample_02_fake_pred_fake_100pct.png")
    if not sample_path.exists():
        sample_path = Path("evaluation_results/gradcam_visualizations_fusion/sample_04_fake_pred_fake_100pct.png")

    full_img = cv2.imread(str(sample_path))
    h, w, c = full_img.shape
    pw = w // 3
    
    input_crop = cv2.cvtColor(full_img[:, :pw], cv2.COLOR_BGR2RGB)
    gt_mask = cv2.cvtColor(full_img[:, pw:2*pw], cv2.COLOR_BGR2RGB)
    heatmap_crop = cv2.cvtColor(full_img[:, 2*pw:], cv2.COLOR_BGR2RGB)

    # Compute actual SRM Filter Residual Map
    gray = cv2.cvtColor(input_crop, cv2.COLOR_RGB2GRAY)
    srm_kernel = np.array([
        [-1,  2, -6,  2, -1],
        [ 2, -8, 12, -8,  2],
        [-6, 12,-12, 12, -6],
        [ 2, -8, 12, -8,  2],
        [-1,  2, -6,  2, -1]
    ], dtype=np.float32) / 12.0
    
    srm_filtered = cv2.filter2D(gray, -1, srm_kernel)
    srm_colored = cv2.applyColorMap(srm_filtered, cv2.COLORMAP_VIRIDIS)
    srm_rgb = cv2.cvtColor(srm_colored, cv2.COLOR_BGR2RGB)

    return input_crop, srm_rgb, heatmap_crop


def main():
    input_crop, srm_rgb, heatmap_crop = get_sample_crops()

    fig = plt.figure(figsize=(15, 8.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Main Header
    ax.text(50, 96, "Dual-Stream Semantic-Frequency Fusion Network Architecture (Proposed Model)",
            ha="center", va="center", fontsize=14, fontweight="bold", color="#1A2B4C")
    ax.text(50, 93.5, "Joint Binary Classification & Pixel-Level Spatial Forgery Localization",
            ha="center", va="center", fontsize=10.5, fontstyle="italic", color="#5D6D7E")

    # -------------------------------------------------------------
    # 1. INPUT STAGE (Left: x=2 to 17, y=28 to 82)
    # -------------------------------------------------------------
    ax.add_patch(patches.FancyBboxPatch((2, 28), 15, 54, boxstyle="round,pad=0.5", ec="#1A2B4C", fc="#F2F4F4", lw=1.8))
    ax.text(9.5, 78, "Input Stage", ha="center", va="center", fontsize=11, fontweight="bold", color="#1A2B4C")

    # Centered Input Sub-Axes
    # In fig coords [left, bottom, width, height], container x=2..17 is roughly left=0.035, width=0.13
    ax_input = fig.add_axes([0.045, 0.36, 0.11, 0.38])
    ax_input.imshow(input_crop)
    ax_input.axis("off")
    ax_input.set_title("Face Crop X\nℝ²²⁴ˣ²²⁴ˣ³", fontsize=9, fontweight="bold", pad=5, color="#1A2B4C")

    # -------------------------------------------------------------
    # 2. STREAM 1: SEMANTIC BRANCH (Top: x=23 to 49, y=60 to 86)
    # -------------------------------------------------------------
    ax.add_patch(patches.FancyBboxPatch((23, 60), 26, 26, boxstyle="round,pad=0.5", ec="#27AE60", fc="#E8F8F5", lw=1.8))
    ax.text(36, 83, "Stream 1: Visual Semantic Branch", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#1E8449")
    
    ax.add_patch(patches.FancyBboxPatch((25, 72), 22, 7.5, boxstyle="round,pad=0.3", ec="#27AE60", fc="#D5F5E3", lw=1.2))
    ax.text(36, 75.75, "CLIP ViT-B/32 Vision Transformer\n(Blocks 1-10 Frozen)", ha="center", va="center", fontsize=8.5, color="#145A32")

    ax.add_patch(patches.FancyBboxPatch((25, 62), 22, 7.5, boxstyle="round,pad=0.3", ec="#27AE60", fc="#ABEBC6", lw=1.2))
    ax.text(36, 65.75, "Top 2 Transformer Blocks (Unfrozen)\nLinear Projection (512 → 256)", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#0E6251")

    # -------------------------------------------------------------
    # 3. STREAM 2: FREQUENCY FORENSIC BRANCH (Bottom: x=23 to 49, y=8 to 52)
    # -------------------------------------------------------------
    ax.add_patch(patches.FancyBboxPatch((23, 8), 26, 44, boxstyle="round,pad=0.5", ec="#2980B9", fc="#EBF5FB", lw=1.8))
    ax.text(36, 49, "Stream 2: Frequency Forensic Branch", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#1F618D")

    # Centered SRM Image
    ax_srm = fig.add_axes([0.252, 0.28, 0.085, 0.15])
    ax_srm.imshow(srm_rgb)
    ax_srm.axis("off")
    ax_srm.set_title("3 SRM Noise Filters\nResidual Map R", fontsize=8, fontweight="bold", pad=4, color="#1F618D")

    # EfficientNet Box
    ax.add_patch(patches.FancyBboxPatch((35, 27), 12, 15, boxstyle="round,pad=0.3", ec="#2980B9", fc="#D4E6F1", lw=1.2))
    ax.text(41, 34.5, "EfficientNet-B0\nBackbone\n+ SE Attention", ha="center", va="center", fontsize=8, fontweight="bold", color="#154360")

    # Compression Gating Module
    ax.add_patch(patches.FancyBboxPatch((25, 11), 22, 12, boxstyle="round,pad=0.3", ec="#D4AC0D", fc="#FEF9E7", lw=1.5))
    ax.text(36, 19, "Adaptive Compression Gating g(X)", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#7D6608")
    ax.text(36, 14, "F_gated = F_spatial × g(X) ∈ ℝ¹²⁸ˣ²⁸ˣ²⁸", ha="center", va="center", fontsize=7.5, color="#7D6608")

    # -------------------------------------------------------------
    # 4. MULTI-DOMAIN FUSION LAYER (Center: x=55 to 71, y=32 to 68)
    # -------------------------------------------------------------
    ax.add_patch(patches.FancyBboxPatch((55, 32), 16, 36, boxstyle="round,pad=0.5", ec="#8E44AD", fc="#F4ECF7", lw=1.8))
    ax.text(63, 64, "Multi-Domain Fusion", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#6C3483")
    
    ax.add_patch(patches.FancyBboxPatch((56.5, 49), 13, 10, boxstyle="round,pad=0.3", ec="#8E44AD", fc="#E8DAEF", lw=1.2))
    ax.text(63, 54, "Concatenation\n[f_sem || v_freq]\n∈ ℝ³⁸⁴", ha="center", va="center", fontsize=8, fontweight="bold", color="#4A235A")

    ax.add_patch(patches.FancyBboxPatch((56.5, 35), 13, 11, boxstyle="round,pad=0.3", ec="#8E44AD", fc="#D7BDE2", lw=1.2))
    ax.text(63, 40.5, "Fusion MLP\nLinear → GELU → Linear\nv_fused ∈ ℝ²⁵⁶", ha="center", va="center", fontsize=8, fontweight="bold", color="#4A235A")

    # -------------------------------------------------------------
    # 5. DUAL TASK OUTPUT HEADS (Right: x=77 to 98)
    # -------------------------------------------------------------
    # Classification Head (Top Right: y=60 to 86)
    ax.add_patch(patches.FancyBboxPatch((77, 60), 21, 26, boxstyle="round,pad=0.5", ec="#C0392B", fc="#FDEDEC", lw=1.8))
    ax.text(87.5, 82, "Head A: Binary Classification", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#922B21")
    ax.text(87.5, 75, "Linear Layer → Sigmoid Activation", ha="center", va="center", fontsize=8.5, color="#7B241C")
    
    # Classification Output Box
    ax.add_patch(patches.FancyBboxPatch((79.5, 62.5), 16, 9.5, boxstyle="round,pad=0.3", ec="#C0392B", fc="#E74C3C", lw=1.5))
    ax.text(87.5, 67.25, "PREDICTION: FAKE\nConfidence Rate: 100.0%", ha="center", va="center", fontsize=9, fontweight="bold", color="white")

    # Localization Head (Bottom Right: y=8 to 52)
    ax.add_patch(patches.FancyBboxPatch((77, 8), 21, 44, boxstyle="round,pad=0.5", ec="#D35400", fc="#FBEEE6", lw=1.8))
    ax.text(87.5, 49, "Head B: Spatial Localization", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#A04000")
    ax.text(87.5, 43, "Conv 1×1 → Upsample → Sigmoid", ha="center", va="center", fontsize=8.5, color="#873600")

    # Centered Heatmap Sub-Axes
    ax_heat = fig.add_axes([0.788, 0.12, 0.14, 0.27])
    ax_heat.imshow(heatmap_crop)
    ax_heat.axis("off")
    ax_heat.set_title("Predicted Forgery Heatmap\nM_pred ∈ ℝ²²⁴ˣ²²⁴", fontsize=8.5, fontweight="bold", pad=5, color="#A04000")

    # -------------------------------------------------------------
    # CONNECTING ARROWS & FLOW PIPELINES
    # -------------------------------------------------------------
    arrow_props = dict(arrowstyle="->", lw=2.2, color="#2C3E50")
    
    # Input to Streams
    ax.annotate("", xy=(23, 73), xytext=(17, 55), arrowprops=arrow_props)
    ax.annotate("", xy=(23, 30), xytext=(17, 45), arrowprops=arrow_props)
    
    # Stream 1 & 2 to Fusion
    ax.annotate("", xy=(55, 53), xytext=(49, 73), arrowprops=dict(arrowstyle="->", lw=2.2, color="#27AE60"))
    ax.annotate("", xy=(55, 43), xytext=(49, 30), arrowprops=dict(arrowstyle="->", lw=2.2, color="#2980B9"))
    
    # Fusion to Classification
    ax.annotate("", xy=(77, 73), xytext=(71, 50), arrowprops=dict(arrowstyle="->", lw=2.2, color="#8E44AD"))
    
    # Stream 2 to Localization
    ax.annotate("", xy=(77, 26), xytext=(49, 17), arrowprops=dict(arrowstyle="->", lw=2.2, color="#D35400"))

    out = Path("paper_figures_600dpi/fig1_best_system_architecture_diagram_600dpi.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved Perfectly Aligned 600 DPI Architecture Diagram: {out}")


if __name__ == "__main__":
    main()
