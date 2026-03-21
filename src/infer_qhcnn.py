from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .config import ImageConfig, Paths
from .dataset import discover_dataset, load_image_record
from .preprocess import preprocess_for_features
from .qhcnn import QHCNNConfig, conv_feature_maps_gray_u8, quantum_features_from_pca
from .qhcnn_tl import TLConfig, extract_tl_features_from_rgb_list
from .onnx_backbone import (
    OnnxBackboneConfig,
    ensure_onnx_model,
    extract_onnx_features_from_rgb_list,
)


@dataclass(frozen=True)
class QHCNNPrediction:
    disease_name: str
    accuracy_percent: float
    confidence_score: float
    per_class_probability: dict[str, float]


def _train_qhcnn_on_the_fly(paths: Paths) -> dict[str, Any]:
    """
    Train a small "hybrid CNN + quantum layer + classical classifier" model and persist it.
    Uses Statevector simulation (no Aer DLL dependency).
    """
    img_cfg = ImageConfig()
    cfg = QHCNNConfig()
    tl_cfg = TLConfig()
    onnx_cfg = OnnxBackboneConfig()

    records = discover_dataset(paths)

    # Use the full dataset (no per-class cap), so all available images contribute.
    from collections import defaultdict

    by_label: dict[str, list] = defaultdict(list)
    for r in records:
        by_label[r.label].append(r)
    rng = np.random.default_rng(cfg.random_state)

    # Try transfer-learning features (much higher accuracy potential).
    # Priority (stability on Windows):
    #   1) OpenCV DNN + ONNX backbone (auto-downloaded)
    #   2) TorchVision backbone (if import works)
    #   3) lightweight conv fallback
    rgbs: list[np.ndarray] = []
    y_list: list[str] = []
    for rec in records:
        rgb = load_image_record(rec)
        rgbs.append(rgb)
        y_list.append(rec.label)

    used_transfer = False
    used_onnx = False
    try:
        model_path = ensure_onnx_model(cfg=onnx_cfg, cache_dir=paths.artifacts_dir / "backbones")
        X = extract_onnx_features_from_rgb_list(rgbs, cfg=onnx_cfg, model_path=model_path, batch_size=32)
        used_transfer = True
        used_onnx = True
    except Exception:
        try:
            X = extract_tl_features_from_rgb_list(rgbs, cfg=tl_cfg)
            used_transfer = True
            used_onnx = False
        except Exception:
            X_list: list[np.ndarray] = []
            for rgb in rgbs:
                pre = preprocess_for_features(rgb, img_cfg=img_cfg)
                feats, _ = conv_feature_maps_gray_u8(pre["gray_u8"])
                X_list.append(feats)
            X = np.stack(X_list, axis=0)

    y_str = np.asarray(y_list)
    le = LabelEncoder()
    y = le.fit_transform(y_str)

    # Use an 80/20 train/test split (20% test).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=cfg.random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Two PCAs:
    # - pca_cls: richer classical embedding for high accuracy
    # - pca_q: compact embedding for quantum circuit inputs
    cls_dim = int(min(128, X_train_s.shape[1]))
    pca_cls = PCA(n_components=cls_dim, random_state=cfg.random_state)
    X_train_cls = pca_cls.fit_transform(X_train_s)
    X_test_cls = pca_cls.transform(X_test_s)

    pca_q = PCA(n_components=cfg.num_qubits, random_state=cfg.random_state)
    X_train_q = pca_q.fit_transform(X_train_s)
    X_test_q = pca_q.transform(X_test_s)

    # Fixed circuit parameters (seeded); treat the circuit as a learnable feature map.
    qc, x_params, theta_params, z_ops = None, None, None, None  # for type-checkers only
    theta = rng.normal(loc=0.0, scale=0.8, size=(cfg.num_layers * (cfg.num_qubits - 1) * 4,)).astype(np.float32)

    Q_train = quantum_features_from_pca(X_train_q, cfg=cfg, theta_values=theta)
    Q_test = quantum_features_from_pca(X_test_q, cfg=cfg, theta_values=theta)

    # Hybrid feature vector: [classical PCA | quantum features]
    H_train = np.concatenate([X_train_cls.astype(np.float32), Q_train.astype(np.float32)], axis=1)
    H_test = np.concatenate([X_test_cls.astype(np.float32), Q_test.astype(np.float32)], axis=1)

    # Strong head. Tune lightly with CV for better accuracy.
    base = SVC(kernel="rbf", probability=True)
    param_grid = {
        "C": [20, 50, 100, 200],
        "gamma": ["scale", 0.01, 0.02, 0.05],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=cfg.random_state)
    grid = GridSearchCV(base, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=0)
    grid.fit(H_train, y_train)
    clf = grid.best_estimator_
    y_pred = clf.predict(H_test)
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    import json

    metrics = {
        "accuracy": acc,
        "precision_weighted": prec,
        "recall_weighted": rec,
        "f1_weighted": f1,
        "num_samples": int(X.shape[0]),
        "num_classes": int(len(le.classes_)),
        "num_qubits": int(cfg.num_qubits),
        "num_layers": int(cfg.num_layers),
        "max_samples_per_class": 0,
        "feature_extractor": ("transfer_learning_onnx" if used_onnx else "transfer_learning_torch") if used_transfer else "lightweight_conv",
        "backbone": onnx_cfg.name if used_onnx else (tl_cfg.backbone if used_transfer else None),
    }
    paths.qhcnn_metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    bundle = {
        "image_config": img_cfg,
        "qhcnn_config": cfg,
        "tl_config": tl_cfg,
        "onnx_config": onnx_cfg,
        "used_transfer": used_transfer,
        "used_onnx": used_onnx,
        "label_encoder": le,
        "scaler": scaler,
        "pca_cls": pca_cls,
        "pca_q": pca_q,
        "theta": theta,
        "classifier": clf,
    }
    joblib.dump(bundle, paths.qhcnn_model_path)
    return bundle


def load_qhcnn_bundle(paths: Paths) -> dict[str, Any]:
    if not paths.qhcnn_model_path.exists():
        return _train_qhcnn_on_the_fly(paths)
    bundle = joblib.load(paths.qhcnn_model_path)
    # Backward-compat: older bundles won't have these keys. Retrain to use improved pipeline.
    if "used_transfer" not in bundle or "tl_config" not in bundle:
        return _train_qhcnn_on_the_fly(paths)
    # If the saved bundle used the weak fallback extractor, retrain with a stronger backbone.
    if not bool(bundle.get("used_transfer", False)):
        return _train_qhcnn_on_the_fly(paths)
    return bundle


def predict_qhcnn(rgb: np.ndarray) -> QHCNNPrediction:
    paths = Paths()
    bundle = load_qhcnn_bundle(paths)

    img_cfg: ImageConfig = bundle["image_config"]
    cfg: QHCNNConfig = bundle["qhcnn_config"]
    tl_cfg: TLConfig = bundle.get("tl_config", TLConfig())
    used_transfer: bool = bool(bundle.get("used_transfer", False))
    onnx_cfg: OnnxBackboneConfig = bundle.get("onnx_config", OnnxBackboneConfig())
    used_onnx: bool = bool(bundle.get("used_onnx", False))
    le: LabelEncoder = bundle["label_encoder"]
    scaler: StandardScaler = bundle["scaler"]
    pca_cls: PCA = bundle["pca_cls"]
    pca_q: PCA = bundle["pca_q"]
    theta: np.ndarray = bundle["theta"]
    clf: SVC = bundle["classifier"]

    if used_transfer and not used_onnx:
        feats = extract_tl_features_from_rgb_list([rgb], cfg=tl_cfg)[0]
    elif used_transfer and used_onnx:
        model_path = ensure_onnx_model(cfg=onnx_cfg, cache_dir=paths.artifacts_dir / "backbones")
        feats = extract_onnx_features_from_rgb_list([rgb], cfg=onnx_cfg, model_path=model_path, batch_size=1)[0]
    else:
        pre = preprocess_for_features(rgb, img_cfg=img_cfg)
        feats, _ = conv_feature_maps_gray_u8(pre["gray_u8"])
    feats_s = scaler.transform(feats.reshape(1, -1))
    x_cls = pca_cls.transform(feats_s)  # shape (1, cls_dim)
    x_q = pca_q.transform(feats_s)  # shape (1, num_qubits)
    q = quantum_features_from_pca(x_q[0], cfg=cfg, theta_values=theta)  # shape (1, num_qubits)
    h = np.concatenate([x_cls.astype(np.float32), q.astype(np.float32)], axis=1)

    probs = clf.predict_proba(h)[0]
    top2 = np.sort(probs)[-2:] if probs.shape[0] >= 2 else probs
    conf = float(top2[-1] - top2[-2]) if top2.shape[0] == 2 else float(top2[-1])

    y_hat = int(np.argmax(probs))
    disease = str(le.inverse_transform([y_hat])[0])

    per_class = {str(le.inverse_transform([i])[0]): float(probs[i]) for i in range(probs.shape[0])}

    acc_percent = float("nan")
    if paths.qhcnn_metrics_path.exists():
        import json

        metrics = json.loads(paths.qhcnn_metrics_path.read_text(encoding="utf-8"))
        if "accuracy" in metrics:
            acc_percent = float(metrics["accuracy"]) * 100.0

    return QHCNNPrediction(
        disease_name=disease,
        accuracy_percent=acc_percent,
        confidence_score=float(conf),
        per_class_probability=per_class,
    )

