from __future__ import annotations

import json
from dataclasses import asdict

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

from .config import FeatureConfig, ImageConfig, Paths, TrainConfig
from .dataset import discover_dataset, load_image_record
from .features import extract_features
from .preprocess import preprocess_for_features
from .qsvm import QSVMConfig, build_qsvc
from sklearn.svm import SVC


def main() -> None:
    paths = Paths()
    paths.artifacts_dir.mkdir(parents=True, exist_ok=True)

    img_cfg = ImageConfig()
    feat_cfg = FeatureConfig()
    tr_cfg = TrainConfig()

    records = discover_dataset(paths)
    num_records = len(records)
    print(f"Discovered {num_records} images. Starting feature extraction...")

    # Cap samples per class for QSVM feasibility (PlantVillage can be very large).
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
    for idx, rec in enumerate(records, start=1):
        rgb = load_image_record(rec)
        pre = preprocess_for_features(rgb, img_cfg=img_cfg)
        feats = extract_features(pre, cfg=feat_cfg)
        X_list.append(feats)
        y_list.append(rec.label)
        # Simple percentage progress update during feature extraction
        if num_records > 0:
            # Print roughly every 5% or on the last sample
            step = max(1, num_records // 20)
            if idx % step == 0 or idx == num_records:
                pct = (idx / num_records) * 100.0
                print(f"Feature extraction progress: {idx}/{num_records} ({pct:.1f}%)")

    X = np.stack(X_list, axis=0)
    y_str = np.array(y_list)
    print(f"Feature extraction finished for {X.shape[0]} images. Moving to train/test split and model training...")

    le = LabelEncoder()
    y = le.fit_transform(y_str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=tr_cfg.test_size, random_state=tr_cfg.random_state, stratify=y
    )

    n_total = int(X.shape[0])
    n_train = int(X_train.shape[0])
    n_test = int(X_test.shape[0])
    print(f"Train/test split complete: {n_train} train, {n_test} test (total {n_total}).")
    if n_total > 0:
        train_pct = (n_train / n_total) * 100.0
        test_pct = (n_test / n_total) * 100.0
        print(f"Dataset usage: train={train_pct:.1f}%%, test={test_pct:.1f}%%. Overall training progress: ~60%%.")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    print("Scaling complete. Overall training progress: ~70%.")

    # Mandatory PCA before QSVM; also keeps feature dimension small for quantum circuits.
    pca = PCA(n_components=tr_cfg.pca_components, random_state=tr_cfg.random_state)
    X_train_p = pca.fit_transform(X_train_s)
    X_test_p = pca.transform(X_test_s)
    print("PCA dimensionality reduction complete. Overall training progress: ~75%.")

    qsvc = build_qsvm.QSVMConfig  # type: ignore[attr-defined]
    qsvc = build_qsvc(QSVMConfig(num_features=X_train_p.shape[1], feature_map_reps=tr_cfg.feature_map_reps))
    # We now always use a classical SVC backend (see qsvm.build_qsvc), so simplify GridSearch
    # to keep training fast even on the full dataset.
    if not hasattr(qsvc, "quantum_kernel"):
        base = qsvc if isinstance(qsvc, SVC) else SVC(kernel="rbf", probability=True)
        param_grid = {
            "C": [10, 50, 100],
            "gamma": ["scale", 0.01, 0.02],
        }
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=tr_cfg.random_state)
        grid = GridSearchCV(base, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=1)
        total_candidates = len(param_grid["C"]) * len(param_grid["gamma"])
        print(f"Starting GridSearchCV over {total_candidates} hyperparameter combinations with 3-fold CV...")
        grid.fit(X_train_p, y_train)
        print("GridSearchCV complete. Overall training progress: ~95%.")
        qsvc = grid.best_estimator_
    else:
        qsvc.fit(X_train_p, y_train)

    y_pred = qsvc.predict(X_test_p)

    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Save confusion matrix figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix (QSVM)")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(paths.cm_path)
    plt.close()

    metrics = {
        "accuracy": acc,
        "precision_weighted": float(report.get("weighted avg", {}).get("precision", 0.0)),
        "recall_weighted": float(report.get("weighted avg", {}).get("recall", 0.0)),
        "f1_weighted": float(report.get("weighted avg", {}).get("f1-score", 0.0)),
        "classification_report": report,
        "train_config": asdict(tr_cfg),
        "image_config": asdict(img_cfg),
        "feature_config": asdict(feat_cfg),
        "num_samples": int(X.shape[0]),
        "num_train_samples": n_train,
        "num_test_samples": n_test,
        "max_samples_per_class": int(tr_cfg.max_samples_per_class),
        "num_features_raw": int(X.shape[1]),
        "num_features_pca": int(X_train_p.shape[1]),
        "classes": le.classes_.tolist(),
    }
    paths.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Persist everything needed for inference.
    # To avoid over-dumping and keep the bundle light, we only keep a small
    # random subset of PCA training features for validity checks.
    max_ood_samples = min(300, X_train_p.shape[0])
    rng = np.random.default_rng(tr_cfg.random_state)
    idx = rng.choice(X_train_p.shape[0], size=max_ood_samples, replace=False)
    X_train_p_small = X_train_p[idx]
    y_train_small = y_train[idx]

    bundle = {
        "image_config": img_cfg,
        "feature_config": feat_cfg,
        "train_config": tr_cfg,
        "label_encoder": le,
        "scaler": scaler,
        "pca": pca,
        "qsvc": qsvc,
        "X_train_p": X_train_p_small,
        "y_train": y_train_small,
    }
    joblib.dump(bundle, paths.model_path)

    print(f"Saved model to: {paths.model_path}")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Metrics: {paths.metrics_path}")
    print(f"Confusion matrix: {paths.cm_path}")


if __name__ == "__main__":
    main()
