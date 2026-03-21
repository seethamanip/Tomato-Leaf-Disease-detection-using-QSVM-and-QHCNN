#!/usr/bin/env python
"""Quick test to verify predictions work on different class images."""

import os
import sys
from pathlib import Path

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import Paths
from src.infer import predict
from src.image_io import load_pil_from_bytes
from PIL import Image
import numpy as np

paths = Paths()
data_root = paths.data_root

# Find one image from each class
class_folders = list(data_root.iterdir())
class_folders = [f for f in class_folders if f.is_dir()]

print(f"Found {len(class_folders)} class folders")
print("-" * 60)

for folder in sorted(class_folders):
    images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
    if images:
        img_path = images[0]
        print(f"\nTesting: {folder.name}")
        print(f"Image: {img_path.name}")
        
        # Load and convert to RGB
        pil = Image.open(img_path).convert("RGB")
        rgb = np.array(pil)
        
        # Predict
        try:
            result, explain_ctx = predict(rgb)
            print(f"Predicted: {result.disease_name}")
            print(f"Confidence: {result.confidence_score:.4f}")
            print(f"Per-class similarities:")
            for cls, sim in sorted(result.per_class_similarity.items()):
                print(f"  {cls}: {sim:.4f}")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

print("\n" + "-" * 60)
print("Test complete!")
