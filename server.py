from __future__ import annotations

import base64
import io
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

ROOT = Path(__file__).resolve().parent
# Add project root to Python path (safe on Windows)
project_root = os.path.abspath(str(ROOT))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.image_io import load_pil_from_bytes, pil_to_numpy_rgb, validate_image_bytes
from src.config import ImageConfig
from src.infer import predict_validated
from src.infer_qhcnn import predict_qhcnn
from src.preprocess import preprocess_for_features
from src.qhcnn import conv_feature_maps_gray_u8
WEB_DIR = ROOT / "web"

app = Flask(__name__, static_folder=None)

def _load_metrics_json(path: Path) -> dict[str, Any] | None:
    try:
        if path.exists():
            import json

            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _extract_basic_metrics(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    if not metrics:
        return None
    acc = metrics.get("accuracy", None)
    prec = metrics.get("precision_weighted", None)
    rec = metrics.get("recall_weighted", None)
    f1 = metrics.get("f1_weighted", None)
    if acc is None and prec is None and rec is None and f1 is None:
        # Try derive from classification_report if present
        rep = metrics.get("classification_report", None)
        if isinstance(rep, dict):
            w = rep.get("weighted avg", {}) if isinstance(rep.get("weighted avg", {}), dict) else {}
            prec = w.get("precision", prec)
            rec = w.get("recall", rec)
            f1 = w.get("f1-score", f1)
            acc = metrics.get("accuracy", acc)
    out: dict[str, Any] = {}
    if acc is not None:
        out["accuracy"] = float(acc)
    if prec is not None:
        out["precision"] = float(prec)
    if rec is not None:
        out["recall"] = float(rec)
    if f1 is not None:
        out["f1"] = float(f1)

    # Dataset usage details (if present)
    n_total = metrics.get("num_samples", None)
    n_train = metrics.get("num_train_samples", None)
    n_test = metrics.get("num_test_samples", None)
    if n_total is not None:
        out["num_samples"] = int(n_total)
    if n_train is not None:
        out["num_train_samples"] = int(n_train)
    if n_test is not None:
        out["num_test_samples"] = int(n_test)
    if n_total and n_train:
        out["train_fraction"] = float(n_train) / float(n_total)
    if n_total and n_test:
        out["test_fraction"] = float(n_test) / float(n_total)
    return out if out else None


def _load_model_metrics_with_fallback(*, primary: Path, fallback: Path) -> dict[str, float] | None:
    """
    Load and merge metrics from the primary path and the fallback path.
    This ensures that if the primary path (eval metrics) has precision/recall
    but lacks num_samples, we still pull num_samples from the fallback path.
    """
    m_primary = _extract_basic_metrics(_load_metrics_json(primary))
    m_fallback = _extract_basic_metrics(_load_metrics_json(fallback))
    
    if not m_primary and not m_fallback:
        return None
        
    merged = {}
    if m_fallback:
        merged.update(m_fallback)
    if m_primary:
        merged.update(m_primary)
        
    return merged


def _png_data_url_from_rgb_u8(rgb: np.ndarray) -> str:
    img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _png_data_url_from_gray_u8(gray: np.ndarray) -> str:
    img = Image.fromarray(gray.astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/assets/<path:filename>")
def assets(filename: str):
    return send_from_directory(WEB_DIR, filename)


@app.post("/api/predict")
def api_predict():
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "Missing 'image' file field."}), 400

    f = request.files["image"]
    image_bytes = f.read()

    img_cfg = ImageConfig()
    try:
        validate_image_bytes(image_bytes, cfg=img_cfg)
        pil = load_pil_from_bytes(image_bytes)
        rgb = pil_to_numpy_rgb(pil)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    # Prediction with validity gate (refuse to predict on non-leaf uploads)
    qsvm_pred = predict_validated(rgb)

    # Build "frame-by-frame" pipeline visuals (preprocess + extracted signals)
    pre = preprocess_for_features(rgb, img_cfg=img_cfg)
    qsvm_steps: list[dict[str, Any]] = []
    qsvm_steps.append({"title": "Resized + Denoised (RGB)", "image": _png_data_url_from_rgb_u8(pre["rgb_u8"])})
    qsvm_steps.append({"title": "Grayscale", "image": _png_data_url_from_gray_u8(pre["gray_u8"])})

    # Edges (recompute quickly here to visualize)
    import cv2

    edges = cv2.Canny(pre["gray_u8"], threshold1=50, threshold2=150)
    qsvm_steps.append({"title": "Edges (Canny)", "image": _png_data_url_from_gray_u8(edges)})

    # HSV channels (normalized) as grayscale images for understanding feature extraction
    hsv = pre["hsv_float"]
    h_u8 = np.clip(hsv[..., 0] * 255.0, 0, 255).astype(np.uint8)
    s_u8 = np.clip(hsv[..., 1] * 255.0, 0, 255).astype(np.uint8)
    v_u8 = np.clip(hsv[..., 2] * 255.0, 0, 255).astype(np.uint8)
    qsvm_steps.append({"title": "HSV: H channel", "image": _png_data_url_from_gray_u8(h_u8)})
    qsvm_steps.append({"title": "HSV: S channel", "image": _png_data_url_from_gray_u8(s_u8)})
    qsvm_steps.append({"title": "HSV: V channel", "image": _png_data_url_from_gray_u8(v_u8)})

    # QHCNN visual steps: pooled classical conv feature maps
    _, pooled_vis = conv_feature_maps_gray_u8(pre["gray_u8"])
    qhcnn_steps: list[dict[str, Any]] = []
    titles = ["Conv map: Sobel X", "Conv map: Sobel Y", "Conv map: Laplacian", "Conv map: Sharpen"]
    for t, im in zip(titles, pooled_vis):
        qhcnn_steps.append({"title": t, "image": _png_data_url_from_gray_u8(im)})

    payload: dict[str, Any] = {
        "ok": True,
        "valid": bool(qsvm_pred.is_valid_image),
        "invalid_reason": qsvm_pred.invalid_reason,
        "validity_scores": qsvm_pred.validity_scores,
        "steps": {"qsvm": qsvm_steps, "qhcnn": qhcnn_steps},
    }

    # Attach performance metrics (from saved artifacts) for both models
    from src.config import Paths

    paths = Paths()
    qsvm_m = _load_model_metrics_with_fallback(
        primary=paths.artifacts_dir / "qsvm_eval_metrics.json",
        fallback=paths.metrics_path,
    )
    qhcnn_m = _load_model_metrics_with_fallback(
        primary=paths.qhcnn_metrics_path,
        fallback=paths.artifacts_dir / "qhcnn_eval_metrics.json",
    )
    payload["metrics"] = {"qsvm": qsvm_m, "qhcnn": qhcnn_m}

    if qsvm_pred.is_valid_image and qsvm_pred.result is not None:
        qhcnn_res = predict_qhcnn(rgb)
        
        # Load optimized ensemble weights if they exist
        out_ensemble = None
        weights_path = paths.artifacts_dir / "ensemble_weights.json"
        
        if weights_path.exists():
            try:
                import json
                w_data = json.loads(weights_path.read_text(encoding="utf-8"))
                w_qsvm = w_data.get("qsvm_weight", 0.5)
                w_qhcnn = w_data.get("qhcnn_weight", 0.5)
                
                # Align probabilities
                classes = set(qsvm_pred.result.per_class_similarity.keys()).union(
                    qhcnn_res.per_class_probability.keys()
                )
                
                def softmax(x_dict):
                    items = list(x_dict.items())
                    keys = [k for k,v in items]
                    vals = np.array([v for k,v in items])
                    if len(vals) == 0: return {}
                    e_x = np.exp(vals - np.max(vals))
                    probs = e_x / e_x.sum()
                    return dict(zip(keys, probs))
                
                q_probs = softmax(qsvm_pred.result.per_class_similarity)
                h_probs = qhcnn_res.per_class_probability
                
                ensemble_probs = {}
                for c in classes:
                    ensemble_probs[c] = (
                        w_qsvm * q_probs.get(c, 0.0) +
                        w_qhcnn * h_probs.get(c, 0.0)
                    )
                
                best_cls = max(ensemble_probs.items(), key=lambda x: x[1])
                out_ensemble = {
                    "disease_name": best_cls[0],
                    "confidence_score": float(best_cls[1]),
                    "per_class": ensemble_probs,
                    "weights": {"qsvm": w_qsvm, "qhcnn": w_qhcnn}
                }
            except Exception as e:
                print(f"Error computing ensemble: {e}")
        
        payload["predictions"] = {
            "qsvm": {
                "disease_name": qsvm_pred.result.disease_name,
                "accuracy_percent": qsvm_pred.result.accuracy_percent,
                "confidence_score": qsvm_pred.result.confidence_score,
                "per_class": qsvm_pred.result.per_class_similarity,
            },
            "qhcnn": {
                "disease_name": qhcnn_res.disease_name,
                "accuracy_percent": qhcnn_res.accuracy_percent,
                "confidence_score": qhcnn_res.confidence_score,
                "per_class": qhcnn_res.per_class_probability,
            },
            "ensemble": out_ensemble
        }
    else:
        payload["predictions"] = {"qsvm": None, "qhcnn": None, "ensemble": None}

    return jsonify(payload)



@app.get("/api/metrics")
def api_metrics():
    """Return saved performance metrics for both models."""
    from src.config import Paths
    paths = Paths()
    qsvm_m = _load_model_metrics_with_fallback(
        primary=paths.artifacts_dir / "qsvm_eval_metrics.json",
        fallback=paths.metrics_path,
    )
    qhcnn_m = _load_model_metrics_with_fallback(
        primary=paths.qhcnn_metrics_path,
        fallback=paths.artifacts_dir / "qhcnn_eval_metrics.json",
    )
    return jsonify({"ok": True, "qsvm": qsvm_m, "qhcnn": qhcnn_m})


if __name__ == "__main__":
    # Visit: http://127.0.0.1:5000
    # Disable auto-reloader to avoid connection resets during long requests (e.g., first QHCNN training).
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
