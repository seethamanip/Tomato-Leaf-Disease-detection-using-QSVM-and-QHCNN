from __future__ import annotations

import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops

from .config import FeatureConfig


def _hsv_hist_features(hsv_float: np.ndarray, bins: int) -> np.ndarray:
    # hsv_float in [0,1]
    feats: list[np.ndarray] = []
    for c in range(3):
        hist, _ = np.histogram(hsv_float[..., c].ravel(), bins=bins, range=(0.0, 1.0), density=True)
        feats.append(hist.astype(np.float32))
    return np.concatenate(feats, axis=0)


def _edge_features(gray_u8: np.ndarray, low: int, high: int) -> np.ndarray:
    edges = cv2.Canny(gray_u8, threshold1=low, threshold2=high)
    # summarize edges with simple statistics
    edge_density = float((edges > 0).mean())
    # projection profiles (coarse) help capture spot distributions
    row_sum = (edges > 0).mean(axis=1)
    col_sum = (edges > 0).mean(axis=0)
    # downsample projections to keep feature count manageable
    row_ds = cv2.resize(row_sum.astype(np.float32).reshape(-1, 1), (1, 16), interpolation=cv2.INTER_AREA).ravel()
    col_ds = cv2.resize(col_sum.astype(np.float32).reshape(1, -1), (16, 1), interpolation=cv2.INTER_AREA).ravel()
    return np.concatenate([np.array([edge_density], dtype=np.float32), row_ds, col_ds], axis=0)


def _glcm_features(gray_u8: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    # Quantize grayscale to cfg.glcm_levels to stabilize GLCM and reduce noise.
    q = np.floor(gray_u8.astype(np.float32) / 256.0 * cfg.glcm_levels).astype(np.uint8)
    q = np.clip(q, 0, cfg.glcm_levels - 1)
    glcm = graycomatrix(
        q,
        distances=list(cfg.glcm_distances),
        angles=list(cfg.glcm_angles),
        levels=cfg.glcm_levels,
        symmetric=True,
        normed=True,
    )
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    feat_list: list[np.ndarray] = []
    for p in props:
        v = graycoprops(glcm, p)  # shape: (len(distances), len(angles))
        feat_list.append(v.astype(np.float32).ravel())
    return np.concatenate(feat_list, axis=0)


def extract_features(pre: dict[str, np.ndarray], cfg: FeatureConfig) -> np.ndarray:
    """
    Combines:
      - HSV hist (color)
      - GLCM (texture)
      - Canny edge summary
    Returns a 1D float vector.
    """
    hsv = pre["hsv_float"]
    gray = pre["gray_u8"]
    f_color = _hsv_hist_features(hsv, bins=cfg.hsv_bins)
    f_tex = _glcm_features(gray, cfg=cfg)
    f_edge = _edge_features(gray, low=cfg.canny_low, high=cfg.canny_high)
    feats = np.concatenate([f_color, f_tex, f_edge], axis=0).astype(np.float32)
    # replace any nan/inf
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


def feature_names(cfg: FeatureConfig) -> list[str]:
    names: list[str] = []
    # HSV hist
    for c, cname in enumerate(["H", "S", "V"]):
        for b in range(cfg.hsv_bins):
            names.append(f"hsv_{cname}_bin_{b}")
    # GLCM
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    for p in props:
        for d in cfg.glcm_distances:
            for a_idx in range(len(cfg.glcm_angles)):
                names.append(f"glcm_{p}_d{d}_a{a_idx}")
    # edge
    names.append("edge_density")
    for i in range(16):
        names.append(f"edge_rowproj_{i}")
    for i in range(16):
        names.append(f"edge_colproj_{i}")
    return names

