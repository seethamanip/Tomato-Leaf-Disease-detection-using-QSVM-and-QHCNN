from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import FOLDER_TO_CLASS, ImageConfig, Paths
from .image_io import pil_to_numpy_rgb


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: str


def discover_dataset(paths: Paths) -> list[ImageRecord]:
    root = paths.data_root
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    records: list[ImageRecord] = []
    # PlantVillage often uses nested folders like:
    # data/tomato/train/Tomato___Early_blight/*.jpg
    # data/tomato/val/Tomato___Early_blight/*.jpg
    # So we search for any directory whose name matches a known class folder.
    candidate_dirs = [p for p in root.rglob("*") if p.is_dir() and p.name in FOLDER_TO_CLASS]
    # Also handle the simple case: class folders directly under root
    candidate_dirs.extend([p for p in root.iterdir() if p.is_dir() and p.name in FOLDER_TO_CLASS])

    # De-duplicate while preserving deterministic order
    seen: set[Path] = set()
    class_dirs: list[Path] = []
    for d in sorted(candidate_dirs):
        if d not in seen:
            seen.add(d)
            class_dirs.append(d)

    for folder in class_dirs:
        label = FOLDER_TO_CLASS[folder.name]
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            for img_path in folder.glob(ext):
                records.append(ImageRecord(path=img_path, label=label))

    if len(records) == 0:
        raise FileNotFoundError(
            f"No images found under {root}. Check folder names and supported extensions."
        )
    return records


def load_image_record(rec: ImageRecord) -> np.ndarray:
    img = Image.open(rec.path).convert("RGB")
    return pil_to_numpy_rgb(img)

