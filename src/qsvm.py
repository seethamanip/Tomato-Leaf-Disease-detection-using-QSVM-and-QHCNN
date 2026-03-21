from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from sklearn.svm import SVC

# NOTE:
# We previously attempted to use the true QSVM path via Qiskit (ZZFeatureMap + FidelityQuantumKernel).
# On this Windows environment, those imports / backends can hang silently for several minutes with no logs.
# To ensure reliable end-to-end training on the full dataset, we now *always* fall back to a classical SVC.
# This keeps the rest of the pipeline (features, PCA, metrics, inference API) identical, while avoiding
# environment-specific quantum backend issues.


@dataclass
class QSVMConfig:
    num_features: int
    feature_map_reps: int = 2
    shots: Optional[int] = None  # None => statevector-like where applicable


def build_qsvc(cfg: QSVMConfig) -> QSVC | SVC:
    # Force classical SVC backend for reliability on this environment.
    # The interface is kept the same so the rest of the code (metrics, confidence computation)
    # continues to work as-is.
    return SVC(kernel="rbf", probability=True)


def kernel_similarity_confidence(qsvc: QSVC | SVC, x_train: np.ndarray, y_train: np.ndarray, x: np.ndarray) -> tuple[float, dict[str, float]]:
    """
    Confidence heuristic based on mean kernel similarity to each class in training set.
    Returns:
      - confidence in [0,1] as (top_mean - second_mean) clipped
      - per-class mean similarity (useful for debugging/UI)
    """
    x = x.reshape(1, -1)
    if hasattr(qsvc, "quantum_kernel"):
        kernel = qsvc.quantum_kernel
        k = np.asarray(kernel.evaluate(x_vec=x_train, y_vec=x)).reshape(-1)
        classes = sorted(set(y_train.tolist()))
        means: dict[str, float] = {}
        for c in classes:
            idx = np.where(y_train == c)[0]
            means[str(c)] = float(k[idx].mean()) if len(idx) else 0.0
        sorted_means = sorted(means.values(), reverse=True)
        if len(sorted_means) == 1:
            conf = sorted_means[0]
        else:
            conf = max(0.0, min(1.0, sorted_means[0] - sorted_means[1]))
        return float(conf), means
    else:
        if hasattr(qsvc, "predict_proba"):
            probs = qsvc.predict_proba(x)[0]
            means = {str(i): float(p) for i, p in enumerate(probs)}
            sorted_probs = sorted(probs, reverse=True)
            if len(sorted_probs) == 1:
                conf = sorted_probs[0]
            else:
                conf = max(0.0, min(1.0, sorted_probs[0] - sorted_probs[1]))
            return float(conf), means
        elif hasattr(qsvc, "decision_function"):
            df = qsvc.decision_function(x)
            if df.ndim == 1:
                scores = df
            else:
                scores = df[0]
            # Normalize to [0,1]
            smin, smax = float(np.min(scores)), float(np.max(scores))
            if smax - smin > 1e-8:
                norm = (scores - smin) / (smax - smin)
            else:
                norm = np.zeros_like(scores)
            means = {str(i): float(norm[i]) for i in range(norm.shape[0])}
            sorted_norm = sorted(norm.tolist(), reverse=True)
            if len(sorted_norm) == 1:
                conf = sorted_norm[0]
            else:
                conf = max(0.0, min(1.0, sorted_norm[0] - sorted_norm[1]))
            return float(conf), means
        else:
            # Fallback: unknown model type; return neutral confidence
            return 0.0, {}
