from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class QHCNNConfig:
    num_qubits: int = 8
    num_layers: int = 2
    random_state: int = 42


def conv_feature_maps_gray_u8(gray_u8: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Minimal CNN-style feature extractor without deep learning frameworks.

    Returns:
      - flat feature vector (float32)
      - list of pooled feature maps (uint8) for visualization
    """
    import cv2

    g = gray_u8.astype(np.float32) / 255.0

    kernels: list[tuple[str, np.ndarray]] = [
        ("edge_sobel_x", np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)),
        ("edge_sobel_y", np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)),
        ("laplacian", np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)),
        ("sharpen", np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)),
    ]

    pooled_vis: list[np.ndarray] = []
    feat_parts: list[np.ndarray] = []
    for _, k in kernels:
        fm = cv2.filter2D(g, ddepth=-1, kernel=k)
        fm = np.maximum(fm, 0.0)  # ReLU
        # "Pooling": downsample to 16x16 for compact representation
        pooled = cv2.resize(fm, (16, 16), interpolation=cv2.INTER_AREA)
        feat_parts.append(pooled.reshape(-1))

        v = pooled
        v = v - float(v.min())
        denom = float(v.max() - v.min()) if float(v.max() - v.min()) > 1e-8 else 1.0
        v_u8 = np.clip(v / denom * 255.0, 0, 255).astype(np.uint8)
        pooled_vis.append(v_u8)

    feats = np.concatenate(feat_parts, axis=0).astype(np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats, pooled_vis


def build_qcnn_feature_map(cfg: QHCNNConfig):
    """
    Builds a parameterized QCNN-like circuit and precomputed Z observables.
    Uses only qiskit.quantum_info.Statevector (no Aer dependency).
    """
    from qiskit.circuit import ParameterVector, QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp

    n = int(cfg.num_qubits)
    x_params = ParameterVector("x", n)
    # Each layer uses (n-1) pair-blocks; each block uses 4 parameters.
    theta_params = ParameterVector("th", cfg.num_layers * (n - 1) * 4)

    qc = QuantumCircuit(n)
    for i in range(n):
        qc.ry(np.pi * x_params[i], i)
        qc.rz(np.pi * x_params[i], i)

    def pair_block(qc_: QuantumCircuit, a: int, b: int, t: Iterable):
        t0, t1, t2, t3 = list(t)
        qc_.ry(t0, a)
        qc_.ry(t1, b)
        qc_.cx(a, b)
        qc_.rz(t2, b)
        qc_.cx(b, a)
        qc_.ry(t3, a)

    t_idx = 0
    for _layer in range(int(cfg.num_layers)):
        # Even pairs: (0,1), (2,3), ...
        for a in range(0, n - 1, 2):
            pair_block(qc, a, a + 1, theta_params[t_idx : t_idx + 4])
            t_idx += 4
        # Odd pairs: (1,2), (3,4), ...
        for a in range(1, n - 1, 2):
            pair_block(qc, a, a + 1, theta_params[t_idx : t_idx + 4])
            t_idx += 4

    z_ops: list[SparsePauliOp] = []
    for qi in range(n):
        # Qiskit uses little-endian ordering in Pauli strings; this mapping works for expectation_value.
        s = ["I"] * n
        s[n - 1 - qi] = "Z"
        z_ops.append(SparsePauliOp.from_list([("".join(s), 1.0)]))

    return qc, x_params, theta_params, z_ops


def quantum_features_from_pca(X_pca: np.ndarray, *, cfg: QHCNNConfig, theta_values: np.ndarray) -> np.ndarray:
    from qiskit.quantum_info import Statevector

    qc, x_params, theta_params, z_ops = build_qcnn_feature_map(cfg)

    X_pca = np.asarray(X_pca, dtype=np.float32)
    if X_pca.ndim == 1:
        X_pca = X_pca.reshape(1, -1)
    if X_pca.shape[1] != cfg.num_qubits:
        raise ValueError(f"Expected PCA dim == num_qubits ({cfg.num_qubits}), got {X_pca.shape[1]}")

    theta_values = np.asarray(theta_values, dtype=np.float32).reshape(-1)
    if theta_values.shape[0] != len(theta_params):
        raise ValueError(f"Expected {len(theta_params)} theta params, got {theta_values.shape[0]}")

    qfeats = np.zeros((X_pca.shape[0], cfg.num_qubits), dtype=np.float32)
    base_bind = {theta_params[i]: float(theta_values[i]) for i in range(len(theta_params))}

    for r in range(X_pca.shape[0]):
        x = np.clip(X_pca[r], -1.0, 1.0)
        bind = dict(base_bind)
        for i in range(cfg.num_qubits):
            bind[x_params[i]] = float(x[i])
        sv = Statevector.from_instruction(qc.assign_parameters(bind, inplace=False))
        for qi, op in enumerate(z_ops):
            qfeats[r, qi] = float(np.real(sv.expectation_value(op)))

    return qfeats

