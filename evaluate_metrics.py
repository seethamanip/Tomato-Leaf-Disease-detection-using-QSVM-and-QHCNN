from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from src.config import Paths
from src.dataset import discover_dataset, load_image_record
from src.infer import predict_validated
from src.infer_qhcnn import predict_qhcnn


def _sample_records_per_class(records, *, max_per_class: int, seed: int):
    by = defaultdict(list)
    for r in records:
        by[r.label].append(r)
    rng = np.random.default_rng(seed)
    out = []
    for label, recs in sorted(by.items(), key=lambda x: x[0]):
        if len(recs) > max_per_class:
            idx = rng.choice(len(recs), size=max_per_class, replace=False)
            out.extend([recs[i] for i in idx])
        else:
            out.extend(recs)
    return out


def main() -> None:
    paths = Paths()
    records = discover_dataset(paths)

    # Keep this fast but representative.
    records = _sample_records_per_class(records, max_per_class=120, seed=42)

    # Evaluate on a held-out split (for metrics only).
    y = np.array([r.label for r in records])
    train_idx, test_idx = train_test_split(
        np.arange(len(records)),
        test_size=0.15,
        random_state=42,
        stratify=y,
    )
    test_recs = [records[i] for i in test_idx]

    y_true: list[str] = []
    y_pred_qsvm: list[str] = []
    y_pred_qhcnn: list[str] = []

    for rec in test_recs:
        rgb = load_image_record(rec)
        y_true.append(rec.label)

        qsvm = predict_validated(rgb)
        if qsvm.is_valid_image and qsvm.result is not None:
            y_pred_qsvm.append(qsvm.result.disease_name)
        else:
            y_pred_qsvm.append("INVALID")

        qh = predict_qhcnn(rgb)
        y_pred_qhcnn.append(qh.disease_name)

    # Filter out invalids for QSVM eval
    mask = [p != "INVALID" for p in y_pred_qsvm]
    y_true_qsvm = [t for t, m in zip(y_true, mask) if m]
    y_pred_qsvm_f = [p for p in y_pred_qsvm if p != "INVALID"]

    def metrics(y_t, y_p):
        return {
            "accuracy": float(accuracy_score(y_t, y_p)),
            "precision_weighted": float(precision_score(y_t, y_p, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_t, y_p, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_t, y_p, average="weighted", zero_division=0)),
            "num_eval_samples": int(len(y_t)),
        }

    qsvm_metrics = metrics(y_true_qsvm, y_pred_qsvm_f) if len(y_true_qsvm) else {}
    qhcnn_metrics = metrics(y_true, y_pred_qhcnn)

    (paths.artifacts_dir / "qsvm_eval_metrics.json").write_text(json.dumps(qsvm_metrics, indent=2), encoding="utf-8")
    (paths.artifacts_dir / "qhcnn_eval_metrics.json").write_text(json.dumps(qhcnn_metrics, indent=2), encoding="utf-8")
    print("Wrote artifacts/qsvm_eval_metrics.json and artifacts/qhcnn_eval_metrics.json")


if __name__ == "__main__":
    main()

