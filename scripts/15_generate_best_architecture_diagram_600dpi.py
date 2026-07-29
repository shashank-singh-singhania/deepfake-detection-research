"""
Script 15 — Generate World-Class 600 DPI System Architecture Diagram for Proposed Model.

Features:
  - High-resolution 600 DPI publication diagram.
  - Embeds actual sample Input Face Crop, SRM Noise Residual Map, and Spatial Localization Heatmap.
  - Custom styled boxes, gradient fills, arrows, mathematical tensor shapes, and component callouts.
  - Output: paper_figures_600dpi/fig1_best_system_architecture_diagram_600dpi.png

Usage:
    python scripts/15_generate_best_architecture_diagram_600dpi.py
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import cv2

DPI = 600


def get_sample_crops():
    """Extracts actual Input Face Crop, SRM Filter Residual, and Heatmap from saved sample."""
    sample_path = Path("evaluation_results/gradcam_visualizations_fusion/sample_04_fake_pred_fake_100pct.png")
    
    if sample_path.exists():
        full_img = cv2.imread(str(sample_path))
        h, w, c = full_img.shape
        # The 3-panel figure has width w. Each panel width is w // 3
        pw = w // 3
        input_crop = cv2.cvtColor(full_img[:, :pw], cv2.COLOR_BGR2RGB)
        gt_mask = cv2.cvtColor(full_img[:, pw:2*pw], cv2.COLOR_BGR2RGB)
        heatmap_crop = cv2.cvtColor(full_img[:, 2*pw:], cv2.COLOR_BGR2RGB)

        # Generate actual SRM Residual map from input_crop
        gray = cv2.cvtColor(input_crop, cv2.COLOR_RGB2GRAY)
        # SRM 5x5 High-pass filter kernel
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
    else:
        # Fallback dummy tensors if file missing
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        return dummy, dummy, dummy


def main():
    input_crop, srm_rgb, heatmap_crop = get_sample_crops()

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Main Header
    ax.text(50, 96, "Dual-Stream Semantic-Frequency Fusion Network Architecture (Proposed Model)",
            ha="center", va="center", fontsize=14, fontweight="bold", color="#1A2B4C")
    ax.text(50, 93.5, "Joint Binary Classification & Pixel-Level Spatial Forgery Localization",
            ha="center", va="center", fontsize=10.5, fontstyle="italic", color="#5D6D7E")

    # -------------------------------------------------------------
    # 1. INPUT STAGE (Left)
    # -------------------------------------------------------------
    # Container Box
    ax.add_patch(patches.FancyBboxPatch((1, 32), 15, 38, boxstyle="round,pad=0.5", ec="#1A2B4C", fc="#F2F4F4", lw=1.8))
    ax.text(8.5, 67, "Input Stage", ha="center", va="center", fontsize=11, fontweight="bold", color="#1A2B4C")
    
    # Embed Input Image
    ax_input = fig.add_axes([0.11, 0.40, 0.10, 0.20]) # [left, bottom, width, height]
    ax_input.imshow(input_crop)
    ax_input.axis("off")
    ax_input.set_title("Face Crop X\nℝ²²⁴ˣ²²⁴ˣ³", fontsize=8.5, fontweight="bold", pad=4)

    # -------------------------------------------------------------
    # 2. STREAM 1: SEMANTIC BRANCH (Top Branch)
    # -------------------------------------------------------------
    ax.add_patch(patches.FancyBboxPatch((20, 60), 28, 28, boxstyle="round,pad=0.5", ec="#27AE60", fc="#E8F8F5", lw=1.8))
    ax.text(34, 85, "Stream 1: Visual Semantic Branch", ha="center", va="center", fontsize=11, fontweight="bold", color="#1E8449")
    
    # Inside Components
    ax.add_patch(patches.FancyBboxPatch((22, 73), 24, 8, boxstyle="round,pad=0.3", ec="#27AE60", fc="#D5F5E3", lw=1.2))
    ax.text(34, 77, "CLIP ViT-B/32 Vision Transformer\n(Blocks 1-10 Frozen)", ha="center", va="center", fontsize=8.5, color="#145A32")

    ax.add_patch(patches.FancyBboxPatch((22, 63), 24, 8, boxstyle="round,pad=0.3", ec="#27AE60", fc="#ABEBC6", lw=1.2))
    ax.text(34, 67, "Top 2 Transformer Blocks (Unfrozen)\nLinear Projection (512 → 256)", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#0E6251")

    # -------------------------------------------------------------
    # 3. STREAM 2: FREQUENCY FORENSIC BRANCH (Bottom Branch)
    # -------------------------------------------------------------
    ax.add_patch(patches.FancyBboxPatch((20, 8), 28, 44, boxstyle="round,pad=0.5", ec="#2980B9", fc="#EBF5FB", lw=1.8))
    ax.text(34, 49, "Stream 2: Frequency Forensic Branch", ha="center", va="center", fontsize=11, fontweight="bold", color="#1F618D")

    # Embed SRM Image
    ax_srm = fig.add_axes([0.245, 0.28, 0.08, 0.14])
    ax_srm.imshow(srm_rgb)
    ax_srm.axis("off")
    ax_srm.set_title("3 SRM Noise Filters\nResidual Map R", fontsize=7.5, fontweight="bold", pad=3)

    # EfficientNet Box
    ax.add_patch(patches.FancyBboxPatch((32, 28), 14, 14, boxstyle="round,pad=0.3", ec="#2980B9", fc="#D4E6F1", lw=1.2))
    ax.text(39, 35, "EfficientNet-B0\nBackbone\n+ SE Attention", ha="center", va="center", fontsize=8, fontweight="bold", color="#154360")

    # Compression Gating Module
    ax.add_patch(patches.FancyBboxPatch((22, 11), 24, 13, boxstyle="round,pad=0.3", ec="#D4AC0D", fc="#FEF9E7", lw=1.5))
    ax.text(34, 20, "Adaptive Compression Gating g(X)", ha="center", va="center", fontsize=9, fontweight="bold", color="#7D6608")
    ax.text(34, 14, "F_gated = F_spatial × g(X) ∈ ℝ¹²⁸ˣ²⁸ˣ²⁸", ha="center", va="center", fontsize=7.5, color="#7D6608")

    # -------------------------------------------------------------
    # 4. MULTI-DOMAIN FUSION LAYER (Center)
    # -------------------------------------------------------------
    ax.add_patch(patches.FancyBboxPatch((52, 33), 18, 35, boxstyle="round,pad=0.5", ec="#8E44AD", fc="#F4ECF7", lw=1.8))
    ax.text(61, 64, "Multi-Domain Fusion", ha="center", va="center", fontsize=11, fontweight="bold", color="#6C3483")
    
    ax.add_patch(patches.FancyBboxPatch((54, 50), 14, 9, boxstyle="round,pad=0.3", ec="#8E44AD", fc="#E8DAEF", lw=1.2))
    ax.text(61, 54.5, "Concatenation\n[f_sem || v_freq]\n∈ ℝ³⁸⁴", ha="center", va="center", fontsize=8, fontweight="bold", color="#4A235A")

    ax.add_patch(patches.FancyBboxPatch((54, 36), 14, 11, boxstyle="round,pad=0.3", ec="#8E44AD", fc="#D7BDE2", lw=1.2))
    ax.text(61, 41.5, "Fusion MLP\nLinear → GELU → Linear\nv_fused ∈ ℝ²⁵⁶", ha="center", va="center", fontsize=8, fontweight="bold", color="#4A235A")

    # -------------------------------------------------------------
    # 5. DUAL TASK OUTPUT HEADS (Right)
    # -------------------------------------------------------------
    # Classification Head (Top Right)
    ax.add_patch(patches.FancyBboxPatch((74, 58), 24, 26, boxstyle="round,pad=0.5", ec="#C0392B", fc="#FDEDEC", lw=1.8))
    ax.text(86, 80, "Head A: Binary Classification", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#922B21")
    ax.text(86, 73, "Linear Layer → Sigmoid Activation", ha="center", va="center", fontsize=8.5, color="#7B241C")
    
    # Classification Prediction Callout Box
    ax.add_patch(patches.FancyBboxPatch((77, 60), 18, 9, boxstyle="round,pad=0.3", ec="#C0392B", fc="#E74C3C", lw=1.5))
    ax.text(86, 64.5, "PREDICTION: FAKE\nConfidence Rate: 100.0%", ha="center", va="center", fontsize=9, fontweight="bold", color="white")

    # Localization Head (Bottom Right)
    ax.add_patch(patches.FancyBboxPatch((74, 8), 24, 45, boxstyle="round,pad=0.5", ec="#D35400", fc="#FBEEE6", lw=1.8))
    ax.text(86, 49, "Head B: Spatial Localization", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#A04000")
    ax.text(86, 43, "Conv 1×1 → Upsample → Sigmoid", ha="center", va="center", fontsize=8.5, color="#873600")

    # Embed Heatmap Output Image
    ax_heat = fig.add_axes([0.765, 0.17, 0.15, 0.22])
    ax_heat.imshow(heatmap_crop)
    ax_heat.axis("off")
    ax_heat.set_title("Predicted Forgery Mask\nM_pred ∈ ℝ²²⁴ˣ²²⁴", fontsize=8.5, fontweight="bold", pad=4, color="#A04000")

    # -------------------------------------------------------------
    # CONNECTING ARROWS & FLOW PIPELINES
    # -------------------------------------------------------------
    arrow_style = dict(arrowstyle="->", lw=2.0, color="#2C3E50")
    
    # Input to Stream 1 & Stream 2
    ax.annotate("", xy=(20, 74), xytext=(16, 52), arrowprops=arrow_style)
    ax.annotate("", xy=(20, 30), xytext=(16, 46), arrowprops=arrow_props_style("#2C3E50"))
    
    # Stream 1 to Fusion
    ax.annotate("", xy=(52, 55), xytext=(48, 74), arrowprops=arrow_props_style("#27AE60"))
    
    # Stream 2 to Fusion & Localization
    ax.annotate("", xy=(52, 42), xytext=(48, 35), arrowprops=arrow_props_style("#2980B9"))
    ax.annotate("", xy=(74, 25), xytext=(48, 20), arrowprops=arrow_props_style("#D35400"))

    # Fusion to Classification Head
    ax.annotate("", xy=(74, 71), xytext=(70, 52), arrowprops=arrow_props_style("#8E44AD"))

    plt.tight_layout()
    out = Path("paper_figures_600dpi/fig1_best_system_architecture_diagram_600dpi.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved Publication Architecture Diagram at 600 DPI: {out}")


def arrow_props_style(col):
    return dict(arrowstyle="->", lw=2.2, color=col)


if __name__ == "__main__":
    main()
