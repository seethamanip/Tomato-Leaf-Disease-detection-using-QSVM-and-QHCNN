from __future__ import annotations

import io
from dataclasses import asdict
from typing import Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

from .config import ImageConfig


def validate_image_bytes(image_bytes: bytes, cfg: ImageConfig) -> None:
    max_bytes = cfg.max_upload_mb * 1024 * 1024
    if len(image_bytes) == 0:
        raise ValueError("Empty file.")
    if len(image_bytes) > max_bytes:
        raise ValueError(f"File too large. Max allowed is {cfg.max_upload_mb} MB.")


def load_pil_from_bytes(image_bytes: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("RGB")
        return img
    except UnidentifiedImageError as e:
        raise ValueError("Unsupported or corrupted image file.") from e


def pil_to_numpy_rgb(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Expected an RGB image.")
    return arr


def get_image_summary(rgb: np.ndarray) -> dict:
    # Minimal stats to pass to Gemini for explanation context (not classification)
    return {
        "shape": tuple(rgb.shape),
        "mean_rgb": [float(x) for x in rgb.reshape(-1, 3).mean(axis=0)],
        "std_rgb": [float(x) for x in rgb.reshape(-1, 3).std(axis=0)],
        "min_rgb": [int(x) for x in rgb.reshape(-1, 3).min(axis=0)],
        "max_rgb": [int(x) for x in rgb.reshape(-1, 3).max(axis=0)],
    }


def debug_config_dict(cfg: ImageConfig) -> dict:
    return asdict(cfg)

