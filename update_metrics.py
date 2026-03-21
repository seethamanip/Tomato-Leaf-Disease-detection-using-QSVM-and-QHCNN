from __future__ import annotations

import json

from src.config import Paths
from src.metrics_utils import extract_weighted_metrics_from_classification_report


def main() -> None:
    paths = Paths()

    # QSVM metrics.json: may or may not have classification_report depending on how it was trained.
    if paths.metrics_path.exists():
        m = json.loads(paths.metrics_path.read_text(encoding="utf-8"))
        if "classification_report" in m and isinstance(m["classification_report"], dict):
            m.update(extract_weighted_metrics_from_classification_report(m["classification_report"]))
        paths.metrics_path.write_text(json.dumps(m, indent=2), encoding="utf-8")
        print(f"Updated QSVM metrics: {paths.metrics_path}")

    # QHCNN metrics are written by its trainer; if missing, we leave as-is.
    if paths.qhcnn_metrics_path.exists():
        m2 = json.loads(paths.qhcnn_metrics_path.read_text(encoding="utf-8"))
        # Nothing to derive unless you add a report; keep existing.
        paths.qhcnn_metrics_path.write_text(json.dumps(m2, indent=2), encoding="utf-8")
        print(f"Rewrote QHCNN metrics: {paths.qhcnn_metrics_path}")


if __name__ == "__main__":
    main()

