## Model Architecture

### Model D: TriConsistencyNet

TriConsistencyNet is a multi-stream network specifically engineered to address the generalization issues of deepfake detectors. Instead of just learning raw representations in isolation, it explicitly models the **consistency** between the spatial and frequency domains.

Below is the complete architectural layout and data flow diagram of TriConsistencyNet:

```text
                            RGB Input Face (B, 3, 224, 224)
                                          │
                      ┌───────────────────┴───────────────────┐
                      ▼                                       ▼
        Spatial Consistency Encoder              Frequency Guidance Encoder (FGE)
             (EfficientNetV2-S)                 (2D FFT Preprocess + Custom CNN)
                      │                                       │
            Spatial Feature Map                      Frequency Feature Map
               (B, 1280, 7, 7)                          (B, 256, 7, 7)
                      │                                       │
                      └───────────────────┬───────────────────┘
                                          ▼
                             Cross-Consistency Attention (CCA)
                                          │
                               Refined Feature Map
                                 (B, 1280, 7, 7)
                                          │
                                          ▼
                              Adaptive Feature Fusion (AFF)
                                          │
                                 Fused Feature Vector
                                      (B, 1280)
                                          │
                                          ▼
                                Dropout & Linear Classifier
                                          │
                                     Final Logits
                                        (B, 2)
```

---

### Step-by-Step Module Specifications

#### 1. Spatial Consistency Encoder
- **Purpose**: Captures visual patterns, semantic facial structures (e.g., eye shape, nose placement), and blending boundaries.
- **Implementation**: Uses a frozen pretrained `EfficientNetV2-S` backbone. Instead of running global pooling immediately, we extract the raw penultimate convolutional feature map:
  $$\text{Input: } (B, 3, 224, 224) \longrightarrow \text{Output: } S \in \mathbb{R}^{B \times 1280 \times 7 \times 7}$$

#### 2. Frequency Guidance Encoder (FGE)
- **Purpose**: Extracts mathematical high-frequency artifacts (such as checkerboard patterns left by GAN decoders or blurring anomalies) that are invisible in the pixel domain.
- **FFT Preprocessing**:
  1. Computes the 2D Fast Fourier Transform (FFT) on the RGB input: $F_{\text{fft}} = \mathcal{F}(x)$.
  2. Extracts the Magnitude Spectrum: $M = |F_{\text{fft}}|$.
  3. Applies Log Compression to reduce the enormous dynamic range: $L = \log(1 + M)$.
  4. Normalizes each sample individually to a $[0, 1]$ range.
- **Frequency CNN**: Passes the normalized spectrum through a lightweight CNN (3 convolution layers and 2 residual blocks, total parameter count $\approx 0.8\text{M}$):
  $$\text{Input: } (B, 3, 224, 224) \longrightarrow \text{Output: } F \in \mathbb{R}^{B \times 256 \times 7 \times 7}$$
- **Resolution Alignment**: If the spatial feature map size differs from the frequency map size, we dynamically resample the frequency map using bilinear interpolation:
  $$F_{\text{aligned}} = \text{Interpolate}(F, \text{size}=\text{shape}(S))$$

#### 3. Cross-Consistency Attention (CCA)
- **Purpose**: Computes a consistency-aware spatial attention map. Traditional two-stream networks concatenate spatial and frequency features, which is redundant. CCA queries frequency artifacts to guide the spatial features.
- **Mathematical Alignment**: 
  We project both streams into a shared 1280-dimensional channel space using $1 \times 1$ convolutions followed by Batch Normalization and SiLU activation:
  $$S_p = \text{BN}(\text{Conv}_{1\times1}(S))$$
  $$F_p = \text{BN}(\text{Conv}_{1\times1}(F_{\text{aligned}}))$$
- **Consistency Interaction**:
  We perform element-wise multiplication ($\odot$) to calculate the cross-domain agreement. Elements are strongly activated *only* where both streams agree:
  $$C = S_p \odot F_p$$
- **Attention Generation**:
  We pass the consistency representation through a gating network to calculate the attention map:
  $$A = \sigma(\text{Conv}_{1\times1}(\text{SiLU}(\text{BN}(\text{Conv}_{1\times1}(C)))))$$
  Where $\sigma$ is the Sigmoid function, mapping values to $[0, 1]$.
- **Residual Refinement**:
  We apply the attention map back to the original spatial features using a residual formulation:
  $$S_{\text{refined}} = S \odot (1.0 + A)$$
  *Note: If $A=0$, the baseline features are preserved, avoiding training collapse.*

#### 4. Adaptive Feature Fusion (AFF)
- **Purpose**: Performs channel-mixing and squeeze-and-excitation to select the most reliable feature channels for classification.
- **Process**:
  1. Passes $S_{\text{refined}}$ through a $1 \times 1$ convolution layer to allow cross-channel interaction.
  2. Applies Global Average Pooling (GAP) to collapse spatial dimensions $(7 \times 7 \longrightarrow 1 \times 1)$, yielding a 1280-dimensional vector.
  3. Feeds the vector through a bottleneck MLP (reducing channels to $320$, then expanding back to $1280$) to learn a channel gate $G \in [0, 1]^{B \times 1280}$.
  4. Calibrates the feature vector:
     $$\mathbf{v}_{\text{fused}} = \text{GAP}(\text{SiLU}(\text{BN}(\text{Conv}_{1\times1}(S_{\text{refined}})))) \odot G$$
  5. Feeds $\mathbf{v}_{\text{fused}}$ through a Dropout layer ($0.3$) and a final Linear classifier to obtain the prediction logits:
     $$\text{Logits} = W \mathbf{v}_{\text{fused}} + \mathbf{b} \in \mathbb{R}^{B \times 2}$$

---

## How the Model is Trained

Deep learning models learn by adjusting their internal parameters (weights) based on a loss function.

- **Class Balancing**: Because our dataset has more fake images than real images, we compute **class weights** (REAL: 3.80, FAKE: 0.58) and apply them to our loss function (`CrossEntropyLoss`). This forces the model to pay equal attention to both classes.
- **Optimizer (`AdamW`)**: The algorithm that updates the weights based on the calculated gradients.
- **Learning Rate Scheduler (`CosineAnnealingLR`)**: Gradually decreases the learning rate over 30 epochs using a cosine wave to stabilize convergence.
- **Mixed Precision**: Uses FP16 computations on the GPU (NVIDIA A100) to speed up training and save memory without losing accuracy.

---