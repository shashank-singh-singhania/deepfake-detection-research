"""
Self-Blended Images (SBI) augmentation — Shiohara & Yamasaki, CVPR 2022.

Core idea: generate a pseudo-fake training pair from a SINGLE real image (no
existing forgery dataset needed for this augmentation) by:
  1. Detecting facial landmarks on the real image.
  2. Building a soft blending mask from the landmark convex hull (face region).
  3. Creating a "source" copy with asymmetric color/quality transforms + a
     slight geometric misalignment (this asymmetry + misalignment is what
     produces a detectable blending-boundary artifact, similar in spirit to
     Face X-ray, but self-blended rather than needing a second identity).
  4. Blending source into the original ("target") within the soft mask at a
     randomized blend ratio.

This is a from-scratch reimplementation following the paper's described
recipe (not a copy of the official repo), intended as a faithful baseline
for comparison — see docs/literature_review_deepfake.xlsx (row: SBI, CVPR 2022).

Landmark detection is behind a small interface (`LandmarkDetector`) so the
blending logic itself is testable with a mock detector (see tests/test_sbi.py)
without needing dlib or a downloaded model file. `DlibLandmarkDetector` is the
provided real implementation — see README.md for how to get the model file.
"""
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Landmark detector interface
# ---------------------------------------------------------------------------
class LandmarkDetector:
    """Abstract interface: implementations return an (N, 2) array of (x, y)
    landmark points in image pixel coordinates, or None if no face found."""

    def get_landmarks(self, rgb_image: np.ndarray) -> Optional[np.ndarray]:
        raise NotImplementedError


class DlibLandmarkDetector(LandmarkDetector):
    """
    Real landmark detector using dlib's 68-point predictor.

    Setup (run once on your DGX):
        pip install dlib
        # download the model file (see README.md Phase 3b-SBI section for the link)
        # then: DlibLandmarkDetector("/path/to/shape_predictor_68_face_landmarks.dat")
    """

    def __init__(self, predictor_path: str):
        import dlib  # imported lazily so the rest of the module works without dlib installed
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def get_landmarks(self, rgb_image: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        faces = self.detector(gray, 1)
        if len(faces) == 0:
            return None
        shape = self.predictor(gray, faces[0])
        pts = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float32)
        return pts


# ---------------------------------------------------------------------------
# Blending core (pure numpy/opencv, detector-agnostic — this is what's tested)
# ---------------------------------------------------------------------------
def build_blend_mask(image_shape: tuple, landmarks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Soft face mask from the landmark convex hull, with random dilation and
    Gaussian-blurred edges (so the blend boundary isn't a hard edge)."""
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(landmarks.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)

    # random dilation for mask-shape diversity across samples
    kernel_size = int(rng.integers(5, 25))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.dilate(mask, kernel, iterations=1)

    # soften edges
    blur_size = int(rng.integers(5, 31))
    if blur_size % 2 == 0:
        blur_size += 1
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

    return (mask.astype(np.float32) / 255.0)


def random_color_transform(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Asymmetric color/quality degradation applied to the 'source' copy only,
    so it's visually distinguishable enough from the target to leave a trace
    once blended (this asymmetry is essential — blending two identical copies
    leaves nothing to detect)."""
    out = img.astype(np.float32)

    # brightness/contrast jitter
    alpha = rng.uniform(0.85, 1.15)   # contrast
    beta = rng.uniform(-20, 20)       # brightness
    out = out * alpha + beta

    out = np.clip(out, 0, 255).astype(np.uint8)

    # random blur or sharpen
    if rng.random() < 0.5:
        k = int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(out, (k, k), 0)
    else:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        out = cv2.filter2D(out, -1, kernel)

    # random downsample/upsample (simulates recompression artifacts)
    if rng.random() < 0.5:
        h, w = out.shape[:2]
        scale = rng.uniform(0.5, 0.9)
        small = cv2.resize(out, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_LINEAR)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    return out


def random_affine_misalign(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Small random affine warp (translate/scale/rotate a few pixels/degrees),
    simulating the slight source-target misalignment that produces a visible
    blending-boundary artifact once composited back onto the target."""
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    angle = rng.uniform(-3, 3)
    scale = rng.uniform(0.97, 1.03)
    tx = rng.uniform(-0.02, 0.02) * w
    ty = rng.uniform(-0.02, 0.02) * h

    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    warped = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return warped


def generate_sbi_sample(rgb_image: np.ndarray, landmarks: np.ndarray,
                         rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Given a real RGB image and its facial landmarks, produce one self-blended
    pseudo-fake image. Returns an RGB uint8 array of the same shape as input.
    """
    rng = rng or np.random.default_rng()

    source = random_color_transform(rgb_image, rng)
    source = random_affine_misalign(source, rng)

    mask = build_blend_mask(rgb_image.shape, landmarks, rng)
    blend_ratio = rng.uniform(0.5, 1.0)  # partial-strength blending, like the paper's lambda
    mask = mask * blend_ratio

    mask_3ch = mask[..., None]
    blended = mask_3ch * source.astype(np.float32) + (1 - mask_3ch) * rgb_image.astype(np.float32)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return blended
