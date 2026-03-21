from __future__ import annotations

from typing import Any


def extract_weighted_metrics_from_classification_report(report: dict[str, Any]) -> dict[str, float]:
    """
    Extract weighted avg precision/recall/f1 from sklearn's classification_report(output_dict=True).
    """
    w = report.get("weighted avg", {})
    if not isinstance(w, dict):
        return {}
    out = {}
    if "precision" in w:
        out["precision_weighted"] = float(w["precision"])
    if "recall" in w:
        out["recall_weighted"] = float(w["recall"])
    if "f1-score" in w:
        out["f1_weighted"] = float(w["f1-score"])
    return out

