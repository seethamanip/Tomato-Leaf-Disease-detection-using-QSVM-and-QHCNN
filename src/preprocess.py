from __future__ import annotations

import cv2
import numpy as np

from .config import FeatureConfig, ImageConfig


def resize_rgb(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    # size is (W,H) for cv2, but we store (H,W) conventionally; keep consistent.
    h, w = size
    resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    return resized


def normalize_float(rgb: np.ndarray) -> np.ndarray:
    return (rgb.astype(np.float32) / 255.0).clip(0.0, 1.0)


def denoise_rgb(rgb: np.ndarray) -> np.ndarray:
    # Mild denoise to reduce sensor noise while preserving edges.
    return cv2.GaussianBlur(rgb, (3, 3), 0)


def rgb_to_hsv_float(rgb_float: np.ndarray) -> np.ndarray:
    rgb_u8 = (rgb_float * 255.0).astype(np.uint8)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    # normalize HSV channels to [0,1] for consistent feature scaling
    hsv[..., 0] /= 179.0
    hsv[..., 1] /= 255.0
    hsv[..., 2] /= 255.0
    return hsv


def rgb_to_gray_u8(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def preprocess_for_features(rgb: np.ndarray, img_cfg: ImageConfig) -> dict[str, np.ndarray]:
    """
    Returns:
      - rgb_u8_resized_denoised
      - rgb_float_norm
      - hsv_float
      - gray_u8
    """
    rgb_resized = resize_rgb(rgb, img_cfg.size)
    rgb_resized = denoise_rgb(rgb_resized)
    rgb_float = normalize_float(rgb_resized)
    hsv_float = rgb_to_hsv_float(rgb_float)
    gray_u8 = rgb_to_gray_u8(rgb_resized)
    return {
        "rgb_u8": rgb_resized,
        "rgb_float": rgb_float,
        "hsv_float": hsv_float,
        "gray_u8": gray_u8,
    }

