from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np

from .config import Paths, FeatureConfig, ImageConfig, TrainConfig
from .features import extract_features
from .image_io import get_image_summary
from .preprocess import preprocess_for_features
from .qsvm import kernel_similarity_confidence, QSVMConfig, build_qsvc
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .dataset import discover_dataset, load_image_record
from .validity import leaf_validity_from_bundle, ValidityResult


@dataclass(frozen=True)
class PredictionResult:
    disease_name: str
    accuracy_percent: float  # from held-out test run (if available)
    confidence_score: float
    per_class_similarity: dict[str, float]


@dataclass(frozen=True)
class ValidatedPrediction:
    is_valid_image: bool
    invalid_reason: str
    validity_scores: dict[str, float]
    result: PredictionResult | None
    explain_ctx: dict | None


def _train_on_the_fly(paths: Paths) -> dict[str, Any]:
    img_cfg = ImageConfig()
    feat_cfg = FeatureConfig()
    tr_cfg = TrainConfig()
    records = discover_dataset(paths)
    if tr_cfg.max_samples_per_class and tr_cfg.max_samples_per_class > 0:
        from collections import defaultdict
        by_label: dict[str, list] = defaultdict(list)
        for r in records:
            by_label[r.label].append(r)
        capped: list = []
        rng = np.random.default_rng(tr_cfg.random_state)
        for label, recs in sorted(by_label.items(), key=lambda x: x[0]):
            if len(recs) > tr_cfg.max_samples_per_class:
                idx = rng.choice(len(recs), size=tr_cfg.max_samples_per_class, replace=False)
                capped.extend([recs[i] for i in idx])
            else:
                capped.extend(recs)
        records = capped
    X_list: list[np.ndarray] = []
    y_list: list[str] = []
    for rec in records:
        rgb = load_image_record(rec)
        pre = preprocess_for_features(rgb, img_cfg=img_cfg)
        feats = extract_features(pre, cfg=feat_cfg)
        X_list.append(feats)
        y_list.append(rec.label)
    X = np.stack(X_list, axis=0)
    y_str = np.array(y_list)
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=tr_cfg.test_size, random_state=tr_cfg.random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    pca = PCA(n_components=tr_cfg.pca_components, random_state=tr_cfg.random_state)
    X_train_p = pca.fit_transform(X_train_s)
    X_test_p = pca.transform(X_test_s)
    qsvc = build_qsvc(QSVMConfig(num_features=X_train_p.shape[1], feature_map_reps=tr_cfg.feature_map_reps))
    qsvc.fit(X_train_p, y_train)
    y_pred = qsvc.predict(X_test_p)
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
        "num_features_pca": int(X_train_p.shape[1]),
    }
    paths.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    bundle = {
        "image_config": img_cfg,
        "feature_config": feat_cfg,
        "train_config": tr_cfg,
        "label_encoder": le,
        "scaler": scaler,
        "pca": pca,
        "qsvc": qsvc,
        "X_train_p": X_train_p,
        "y_train": y_train,
    }
    joblib.dump(bundle, paths.model_path)
    return bundle

def load_bundle(paths: Paths) -> dict[str, Any]:
    """
    Load a pre-trained QSVM/SVC bundle.

    NOTE: We deliberately do NOT train on the fly here anymore, because that can
    block the UI for several minutes on large datasets. If the bundle is missing,
    the caller should run the offline training script first.
    """
    if not paths.model_path.exists():
        raise RuntimeError(
            "QSVM model bundle not found at "
            f"{paths.model_path}. Run 'python -m src.train_qsvm' or 'run_training.py' "
            "once to train the model before using the UI."
        )
    return joblib.load(paths.model_path)


def predict(rgb: np.ndarray) -> tuple[PredictionResult, dict]:
    """
    Predict disease from RGB image array.
    Note: This function is NOT cached to ensure fresh predictions for each image.
    """
    paths = Paths()
    bundle = load_bundle(paths)

    img_cfg = bundle["image_config"]
    feat_cfg = bundle["feature_config"]
    le = bundle["label_encoder"]
    scaler = bundle["scaler"]
    pca = bundle["pca"]
    qsvc = bundle["qsvc"]
    # Training embeddings are optional; newer bundles may omit them for size/speed.
    X_train_p = np.asarray(bundle.get("X_train_p")) if "X_train_p" in bundle else None
    y_train = np.asarray(bundle.get("y_train")) if "y_train" in bundle else None

    pre = preprocess_for_features(rgb, img_cfg=img_cfg)
    feats = extract_features(pre, cfg=feat_cfg).reshape(1, -1)
    feats_s = scaler.transform(feats)
    feats_p = pca.transform(feats_s)

    y_hat = int(qsvc.predict(feats_p)[0])
    disease = str(le.inverse_transform([y_hat])[0])

    # For classical SVC we only need model probabilities; X_train_p/y_train are ignored.
    conf, per_class_numeric = kernel_similarity_confidence(
        qsvc,
        x_train=X_train_p if X_train_p is not None else np.empty((0, feats_p.shape[1])),
        y_train=y_train if y_train is not None else np.empty((0,), dtype=int),
        x=feats_p[0],
    )
    
    # Convert numeric class indices to class names
    per_class = {}
    for class_idx_str, sim_score in per_class_numeric.items():
        try:
            class_idx = int(class_idx_str)
            class_name = str(le.inverse_transform([class_idx])[0])
            per_class[class_name] = sim_score
        except (ValueError, IndexError):
            per_class[class_idx_str] = sim_score

    # If metrics exist, load accuracy; otherwise report NaN.
    acc_percent = float("nan")
    if paths.metrics_path.exists():
        import json

        metrics = json.loads(paths.metrics_path.read_text(encoding="utf-8"))
        if "accuracy" in metrics:
            acc_percent = float(metrics["accuracy"]) * 100.0

    result = PredictionResult(
        disease_name=disease,
        accuracy_percent=acc_percent,
        confidence_score=float(conf),
        per_class_similarity=per_class,
    )

    explain_ctx = {
        "image_summary": get_image_summary(pre["rgb_u8"]),
        "preprocess": {
            "resize": img_cfg.size,
            "color_space": "HSV hist + grayscale GLCM/edges",
        },
        "feature_vector_dim": int(feats.shape[1]),
        "pca_dim": int(feats_p.shape[1]),
        "kernel_similarity_means": per_class,
    }
    return result, explain_ctx


def predict_validated(rgb: np.ndarray) -> ValidatedPrediction:
    """
    Like `predict`, but refuses to classify when the upload looks out-of-distribution
    (e.g., not a tomato leaf photo).
    """
    paths = Paths()
    bundle = load_bundle(paths)

    img_cfg = bundle["image_config"]
    feat_cfg = bundle["feature_config"]
    le = bundle["label_encoder"]
    scaler = bundle["scaler"]
    pca = bundle["pca"]
    qsvc = bundle["qsvc"]
    X_train_p = bundle.get("X_train_p")
    y_train = bundle.get("y_train")

    pre = preprocess_for_features(rgb, img_cfg=img_cfg)
    feats = extract_features(pre, cfg=feat_cfg).reshape(1, -1)
    feats_s = scaler.transform(feats)
    feats_p = pca.transform(feats_s)

    conf, per_class_numeric = kernel_similarity_confidence(
        qsvc,
        x_train=np.asarray(X_train_p),
        y_train=np.asarray(y_train),
        x=feats_p[0],
    )

    validity: ValidityResult = leaf_validity_from_bundle(
        rgb_u8_resized=pre["rgb_u8"],
        feats_p=feats_p[0],
        confidence_score=float(conf),
        bundle=bundle,
    )

    if not validity.is_valid:
        return ValidatedPrediction(
            is_valid_image=False,
            invalid_reason=validity.reason,
            validity_scores=validity.scores,
            result=None,
            explain_ctx=None,
        )

    # Convert numeric class indices to class names
    per_class = {}
    for class_idx_str, sim_score in per_class_numeric.items():
        try:
            class_idx = int(class_idx_str)
            class_name = str(le.inverse_transform([class_idx])[0])
            per_class[class_name] = sim_score
        except (ValueError, IndexError):
            per_class[class_idx_str] = sim_score

    y_hat = int(qsvc.predict(feats_p)[0])
    disease = str(le.inverse_transform([y_hat])[0])

    # If metrics exist, load accuracy; otherwise report NaN.
    acc_percent = float("nan")
    if paths.metrics_path.exists():
        import json

        metrics = json.loads(paths.metrics_path.read_text(encoding="utf-8"))
        if "accuracy" in metrics:
            acc_percent = float(metrics["accuracy"]) * 100.0

    result = PredictionResult(
        disease_name=disease,
        accuracy_percent=acc_percent,
        confidence_score=float(conf),
        per_class_similarity=per_class,
    )

    explain_ctx = {
        "image_summary": get_image_summary(pre["rgb_u8"]),
        "preprocess": {
            "resize": img_cfg.size,
            "color_space": "HSV hist + grayscale GLCM/edges",
        },
        "feature_vector_dim": int(feats.shape[1]),
        "pca_dim": int(feats_p.shape[1]),
        "kernel_similarity_means": per_class,
        "validity": {"reason": validity.reason, "scores": validity.scores},
    }

    return ValidatedPrediction(
        is_valid_image=True,
        invalid_reason="",
        validity_scores=validity.scores,
        result=result,
        explain_ctx=explain_ctx,
    )
