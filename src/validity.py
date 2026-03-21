from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class ValidityResult:
    is_valid: bool
    reason: str
    scores: dict[str, float]


def _basic_sanity_checks(rgb_u8: np.ndarray) -> Optional[ValidityResult]:
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        return ValidityResult(False, "Expected an RGB image.", {})

    h, w, _ = rgb_u8.shape
    if h < 32 or w < 32:
        return ValidityResult(False, "Image is too small to analyze reliably.", {"h": float(h), "w": float(w)})

    rgb_f = rgb_u8.astype(np.float32) / 255.0
    mean = float(rgb_f.mean())
    std = float(rgb_f.std())
    if std < 0.02:
        return ValidityResult(False, "Image has very low detail (blank/blurred).", {"std": std, "mean": mean})
    if mean < 0.08 or mean > 0.92:
        return ValidityResult(False, "Image is too dark or too bright.", {"std": std, "mean": mean})

    # Very weak heuristic: a leaf photo usually contains some green-dominant pixels.
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    greenish = (g > r + 0.05) & (g > b + 0.05) & (g > 0.20)
    green_ratio = float(greenish.mean())
    if green_ratio < 0.01:
        return ValidityResult(False, "Image does not look like a leaf photo (low green content).", {"green_ratio": green_ratio})

    return None


def leaf_validity_from_bundle(
    *,
    rgb_u8_resized: np.ndarray,
    feats_p: np.ndarray,
    confidence_score: float,
    bundle: dict[str, Any],
) -> ValidityResult:
    """
    Heuristic 'is this a tomato leaf-like image?' gate to avoid predicting on out-of-distribution inputs.

    Uses:
      - Basic image sanity checks
      - Feature-space distance to training distribution (PCA space)
      - Model confidence heuristic
    """
    basic = _basic_sanity_checks(rgb_u8_resized)
    if basic is not None:
        return basic

    feats_p = np.asarray(feats_p, dtype=np.float32).reshape(-1)
    X_train_p = np.asarray(bundle.get("X_train_p"), dtype=np.float32)
    if X_train_p.ndim != 2 or X_train_p.shape[1] != feats_p.shape[0]:
        # If training features aren't available, fall back to confidence-only check.
        if confidence_score < 0.10:
            return ValidityResult(False, "Image is not valid (low confidence).", {"confidence": float(confidence_score)})
        return ValidityResult(True, "ok", {"confidence": float(confidence_score)})

    # Distance to training mean (z-normalized) helps reject non-leaf images.
    mu = X_train_p.mean(axis=0)
    sigma = X_train_p.std(axis=0)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    z = (feats_p - mu) / sigma
    z_norm = float(np.linalg.norm(z))

    # Nearest-neighbor distance is a strong OOD signal for weird uploads.
    nn_dist = float(np.min(np.linalg.norm(X_train_p - feats_p[None, :], axis=1)))

    # Dynamic thresholds from training distribution
    z_norm_train = np.linalg.norm((X_train_p - mu[None, :]) / sigma[None, :], axis=1)
    nn_train = np.min(
        np.linalg.norm(X_train_p[:, None, :] - X_train_p[None, :, :], axis=2) + np.eye(X_train_p.shape[0]) * 1e9,
        axis=1,
    )
    z_thr = float(np.quantile(z_norm_train, 0.995))
    nn_thr = float(np.quantile(nn_train, 0.995))

    scores = {
        "confidence": float(confidence_score),
        "z_norm": z_norm,
        "z_thr": z_thr,
        "nn_dist": nn_dist,
        "nn_thr": nn_thr,
    }

    # Require both "looks in-distribution" AND "not extremely low confidence".
    if confidence_score < 0.08:
        return ValidityResult(False, "Image is not valid (too uncertain).", scores)
    if z_norm > z_thr * 1.25 or nn_dist > nn_thr * 1.25:
        return ValidityResult(False, "Image is not valid (not a leaf / out of dataset).", scores)

    return ValidityResult(True, "ok", scores)

