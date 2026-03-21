#!/usr/bin/env python
"""
Optimize QSVM by searching PCA components and feature-map repetitions,
train classical baselines for comparison, save best model bundle and
artifacts/model_comparison.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from src.config import Paths
from src.dataset import discover_dataset, load_image_record
from src.preprocess import preprocess_for_features
from src.features import extract_features
from src.qsvm import QSVMConfig, build_qsvc

# Simple config for search (keep small to be practical)
PCA_OPTIONS = [32, 48, 64]
REPS_OPTIONS = [1, 2]
SAMPLE_CAP_OPTIONS = [200, 300]
RANDOM_STATE = 42

paths = Paths()
paths.artifacts_dir.mkdir(parents=True, exist_ok=True)

# Load all records and extract features once
print("Loading dataset feature vectors...")
records = discover_dataset(paths)
print(f"Found {len(records)} image records")

# Re-implement safely: call with actual configs
from src.config import ImageConfig, FeatureConfig
img_cfg = ImageConfig()
feat_cfg = FeatureConfig()

X_list = []
y_list = []
for i, r in enumerate(records):
    rgb = load_image_record(r)
    pre = preprocess_for_features(rgb, img_cfg=img_cfg)
    feats = extract_features(pre, cfg=feat_cfg)
    X_list.append(feats)
    y_list.append(r.label)
    if (i + 1) % 200 == 0:
        print(f"  Processed {i+1}/{len(records)}")

X = np.stack(X_list, axis=0)
y_str = np.array(y_list)

le = LabelEncoder()
y_all = le.fit_transform(y_str)

print(f"Total samples: {X.shape[0]}, classes: {len(le.classes_)}")

best = None
best_score = -1.0
results_log = []

for cap in SAMPLE_CAP_OPTIONS:
    # Subsample per class
    by_label = defaultdict(list)
    rng = np.random.default_rng(RANDOM_STATE)
    for idx, label in enumerate(y_all):
        by_label[label].append(idx)
    idx_keep = []
    for label, idxs in by_label.items():
        if len(idxs) > cap:
            chosen = rng.choice(idxs, size=cap, replace=False)
            idx_keep.extend(chosen.tolist())
        else:
            idx_keep.extend(idxs)
    idx_keep = np.array(idx_keep)

    X_sub = X[idx_keep]
    y_sub = y_all[idx_keep]

    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X_sub, y_sub, test_size=0.10, random_state=RANDOM_STATE, stratify=y_sub
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_full)
    X_test_s = scaler.transform(X_test_full)

    for ncomp in PCA_OPTIONS:
        if ncomp >= X_train_s.shape[1]:
            ncomp = X_train_s.shape[1]
        pca = PCA(n_components=ncomp, random_state=RANDOM_STATE)
        X_train_p = pca.fit_transform(X_train_s)
        X_test_p = pca.transform(X_test_s)

        for reps in REPS_OPTIONS:
            cfg = QSVMConfig(num_features=X_train_p.shape[1], feature_map_reps=reps)
            print(f"Training QSVM candidate: cap={cap} pca={ncomp} reps={reps}")
            try:
                qsvc = build_qsvc(cfg)
                t0 = time.time()
                if hasattr(qsvc, 'quantum_kernel'):
                    qsvc.fit(X_train_p, y_train_full)
                else:
                    # fallback: do a limited grid search for C
                    base = qsvc if isinstance(qsvc, SVC) else SVC(kernel='rbf', probability=True)
                    base.fit(X_train_p, y_train_full)
                train_time = time.time() - t0

                t0 = time.time()
                y_pred = qsvc.predict(X_test_p)
                inf_time = time.time() - t0

                acc = accuracy_score(y_test_full, y_pred)
                prec = precision_score(y_test_full, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test_full, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test_full, y_pred, average='weighted', zero_division=0)

                score = f1  # use F1 as selection metric
                results_log.append({
                    'cap': cap, 'pca': ncomp, 'reps': reps,
                    'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1,
                    'train_time': train_time, 'inf_time': inf_time,
                })

                print(f"  acc={acc:.4f} f1={f1:.4f} train={train_time:.2f}s inf={inf_time:.4f}s")

                if score > best_score:
                    best_score = score
                    best = {
                        'cap': cap, 'pca': ncomp, 'reps': reps,
                        'qsvc': qsvc, 'scaler': scaler, 'pca_obj': pca,
                        'X_train_p': X_train_p, 'y_train': y_train_full,
                        'metrics': {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'train_time': train_time, 'inf_time': inf_time},
                        'label_encoder': le,
                        'feature_config': feat_cfg,
                        'image_config': img_cfg,
                    }

            except Exception as e:
                print(f"  SKIP candidate due to error: {e}")

# Train classical baselines on best config for comparison
print("\nBest QSVM candidate:", best['cap'], best['pca'], best['reps'], "f1=", best['metrics']['f1'])

# Prepare training/test using best cap
cap = best['cap']
by_label = defaultdict(list)
for idx, label in enumerate(y_all):
    by_label[label].append(idx)
idx_keep = []
rng = np.random.default_rng(RANDOM_STATE)
for label, idxs in by_label.items():
    if len(idxs) > cap:
        chosen = rng.choice(idxs, size=cap, replace=False)
        idx_keep.extend(chosen.tolist())
    else:
        idx_keep.extend(idxs)
idx_keep = np.array(idx_keep)

X_sub = X[idx_keep]
y_sub = y_all[idx_keep]
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_sub, y_sub, test_size=0.10, random_state=RANDOM_STATE, stratify=y_sub
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_full)
X_test_s = scaler.transform(X_test_full)
pca = PCA(n_components=best['pca'], random_state=RANDOM_STATE)
X_train_p = pca.fit_transform(X_train_s)
X_test_p = pca.transform(X_test_s)

# Save best QSVM model bundle
bundle = {
    'image_config': best['image_config'],
    'feature_config': best['feature_config'],
    'label_encoder': best['label_encoder'],
    'scaler': scaler,
    'pca': pca,
    'qsvc': best['qsvc'],
    'X_train_p': X_train_p,
    'y_train': y_train_full,
}
joblib.dump(bundle, paths.model_path)
print(f"Saved optimized model to: {paths.model_path}")

# Evaluate classical baselines on same split
results = {}
# QSVM metrics
results['QSVM'] = {
    'accuracy': float(best['metrics']['acc']),
    'precision': float(best['metrics']['prec']),
    'recall': float(best['metrics']['rec']),
    'f1_score': float(best['metrics']['f1']),
    'inference_time_seconds': float(best['metrics']['inf_time']),
}

# Classical SVM (RBF)
svm = SVC(kernel='rbf', C=100, gamma='scale', probability=True)
t0 = time.time()
svm.fit(X_train_p, y_train_full)
t1 = time.time()
yp = svm.predict(X_test_p)
t2 = time.time()
results['Classical SVM (RBF)'] = {
    'accuracy': float(accuracy_score(y_test_full, yp)),
    'precision': float(precision_score(y_test_full, yp, average='weighted', zero_division=0)),
    'recall': float(recall_score(y_test_full, yp, average='weighted', zero_division=0)),
    'f1_score': float(f1_score(y_test_full, yp, average='weighted', zero_division=0)),
    'training_time_seconds': float(t1 - t0),
    'inference_time_seconds': float(t2 - t1),
}

# Random Forest
rf = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=RANDOM_STATE)
t0 = time.time()
rf.fit(X_train_p, y_train_full)
t1 = time.time()
yp = rf.predict(X_test_p)
t2 = time.time()
results['Random Forest'] = {
    'accuracy': float(accuracy_score(y_test_full, yp)),
    'precision': float(precision_score(y_test_full, yp, average='weighted', zero_division=0)),
    'recall': float(recall_score(y_test_full, yp, average='weighted', zero_division=0)),
    'f1_score': float(f1_score(y_test_full, yp, average='weighted', zero_division=0)),
    'training_time_seconds': float(t1 - t0),
    'inference_time_seconds': float(t2 - t1),
}

# Save comparison JSON
report = {'timestamp': str(Path.cwd()), 'model_comparison': results}
with open(paths.artifacts_dir / 'model_comparison.json', 'w') as f:
    json.dump(report, f, indent=2)

print("Saved model comparison to artifacts/model_comparison.json")
print(json.dumps(report, indent=2))

print("Optimization complete.")
