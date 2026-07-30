# 🌐 High-Level System Architecture Design: Proposed Dual-Stream Fusion Network

**Project Title:** *Dual-Stream Semantic-Frequency Fusion Network for Explainable Deepfake Detection and Cross-Domain Generalization*  
**Authors:** Shashank Singh and Deepinder Kaur  
**Institution:** KIET Deemed to be University  

---

## 🎯 1. High-Level Design Motivation & Core Problem Solved

Standard deepfake detectors suffer from **"Domain Blindness"** because they rely on single-domain feature extraction:

1. **Pure RGB Appearance Models (e.g. standard Xception):** Memorize background shortcuts and specific GAN compression artifacts. When tested zero-shot on unseen modern AI Diffusion Models (Stable Diffusion, InsightFace), their classification collapses to random guessing (~50% AUC).
2. **Pure Frequency Noise Models (e.g. standard SRM filters):** Rely on subtle high-frequency sensor noise traces. When videos are compressed on social media (JPEG/H.264), high-frequency details are destroyed, causing model collapse.

### 💡 The High-Level Solution:
Our **Proposed Architecture** solves both problems simultaneously through a **Dual-Stream Multi-Modal Synergy**:
- **Stream 1 (Semantic Stream):** Focuses on *High-Level Semantic Anatomy* (eye gaze, lighting consistency, facial symmetry) via **CLIP ViT-B/32 Transformer**, providing an unshakeable baseline immune to compression.
- **Stream 2 (Frequency Stream):** Focuses on *Micro-Level Frequency Noise Artifacts* via **SRM Noise Kernels + EfficientNet-B0 + SE Attention**, capturing subtle mathematical manipulation traces.
- **Adaptive Compression Gate $g(X)$:** Dynamically scales frequency feature weights based on estimated video compression degradation.
- **Dual Output Heads:** Simultaneously outputs binary classification score ($P(\text{Fake})$) and a 2D spatial forgery localization heatmap ($M_{\text{pred}}$).

---

## 🏗️ 2. High-Level Modular Component Architecture

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        MODULE 1: INPUT VIDEO FRAME EXTRACTION & CROP                   │
│   Input Video Stream ──► Sampled Video Frames ──► MTCNN Face Crop (224x224x3 RGB)      │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              MODULE 2: DUAL-STREAM FEATURE EXTRACTION                  │
│                                                                                        │
│   ┌──────────────────────────────────────────┐  ┌──────────────────────────────────┐   │
│   │ Stream 1: Visual Semantic Branch         │  │ Stream 2: Frequency Noise Branch │   │
│   │ • Input: 3 RGB Channels                  │  │ • RGB (3 Ch) -> Grayscale (1 Ch) │   │
│   │ • OpenAI CLIP ViT-B/32 Backbone          │  │ • 3 SRM Filters (K1, K2, K3)     │   │
│   │ • Bottom 10 Blocks Frozen (General Info) │  │ • Produces 3 Noise Channels (R)  │   │
│   │ • Top 2 Blocks Unfrozen (Domain Adapt)   │  │ • EfficientNet-B0 + SE Attention │   │
│   │ • Feature Vector: 256-dim f_semantic     │  │ • Feature Map: 128 x 28 x 28     │   │
│   │                                          │  │   F_spatial                      │   │
│   └────────────────────┬─────────────────────┘  └────────────────┬─────────────────┘   │
└────────────────────────┼─────────────────────────────────────────┼─────────────────────┘
                         │                                         │
                         ▼                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        MODULE 3: ADAPTIVE GATING & FEATURE FUSION                      │
│   • Compression Gate g(X) ──► F_gated = F_spatial * g(X)                              │
│   • Adaptive Average Pooling ──► 128-dim v_freq                                         │
│   • Concatenation ──► [f_semantic || v_freq] (384-dim Vector)                           │
│   • Fusion MLP (Dense -> BatchNorm -> GELU -> Dropout 0.3) ──► 256-dim v_fused          │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              MODULE 4: DUAL TASK OUTPUT HEADS                          │
│                                                                                        │
│   ┌──────────────────────────────────────────┐  ┌──────────────────────────────────┐   │
│   │ Task Head 1: Binary Classifier           │  │ Task Head 2: Spatial Localization│   │
│   │ • Linear Projection (256 -> 1)           │  │ • 1x1 Conv (128 -> 1)            │   │
│   │ • Predicts: Logit y (Real vs Fake)       │  │ • Bilinear Upsample to 224x224   │   │
│   │ • Loss: BCEWithLogits                    │  │ • Predicts: 2D Heatmap M_pred    │   │
│   │                                          │  │ • Loss: Auxiliary Mask BCE       │   │
│   └──────────────────────────────────────────┘  └──────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ 3. Detailed Dataflow: RGB to Noise Channel Pipeline

```text
               Input Video Stream (.mp4 / .avi)
                              │
                    Frame Sampling & MTCNN Crop
                              │
               RGB Input Face Crop (3 Channels)
                    X ∈ R^(3 x 224 x 224)
                              │
       ┌──────────────────────┴──────────────────────┐
       │ (Direct RGB 3-Channel Feed)                 │ (Grayscale Conversion)
       ▼                                             ▼
Stream 1: CLIP ViT-B/32                   Grayscale Image (1 Channel)
 (RGB Color & Visual Semantics)             I_gray ∈ R^(1 x 224 x 224)
       │                                             │
       │                                   Convolved with 3 SRM Kernels
       │                                   K1 (1st Order), K2 (2nd Order), K3 (3rd Order)
       │                                             │
       │                                  3 Noise Residual Channels
       │                                    R ∈ R^(3 x 224 x 224)
       │                                             │
       │                                  Stream 2: EfficientNet-B0 + SE
       │                                   (High-Frequency Artifacts)
       ▼                                             ▼
 256-dim f_semantic                           128x28x28 F_spatial
```

---

## ⚙️ 4. Summary of System Design Advantages

| High-Level Design Feature | System Advantage & Empirical Benefit |
|---|---|
| **Video Frame Sampling & MTCNN Alignment** | Converts continuous video streams into aligned $224 \times 224 \times 3$ RGB face crops. |
| **3 SRM Noise Channels ($R_1, R_2, R_3$)** | Converts grayscale images into 3 high-pass noise residual channels, extracting pixel-level noise artifacts invisible to human eyes. |
| **CLIP Vision-Language Backbone** | Captures global visual semantics; achieves **+5.76% higher zero-shot cross-domain AUC** on unseen Stable Diffusion models. |
| **Squeeze-and-Excitation (SE) Attention** | Dynamically recalibrates frequency feature channels, boosting Pointing Game Localization Accuracy by **+13.29%**. |
| **Adaptive Compression Gate $g(X)$** | Dynamically scales frequency feature maps under compression, ensuring near-zero AUC drop (**-0.02%**) under heavy JPEG compression ($Q=50$). |
| **Auxiliary Spatial Localization Head** | Natively generates 2D forgery heatmaps ($224 \times 224$), achieving **88.84% Pointing Game Acc** and **57.84% Mask IoU**. |
