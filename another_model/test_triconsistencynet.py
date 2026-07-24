"""
Quick sanity test for TriConsistencyNet standalone package.
"""

from pathlib import Path
import sys

import torch

ANOTHER_MODEL_DIR = Path(__file__).resolve().parent
SRC_DIR = ANOTHER_MODEL_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import TriConsistencyNet


def main():
    print("Initializing TriConsistencyNet...")
    model = TriConsistencyNet(freeze_backbone=True)
    print(model.param_summary())

    x = torch.randn(2, 3, 224, 224)
    logits = model(x)

    print()
    print("Input shape  :", x.shape)
    print("Logits shape :", logits.shape)
    print("Attention map:", model.last_attention_map.shape)
    print()
    print("Sanity test passed successfully!")


if __name__ == "__main__":
    main()
