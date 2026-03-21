from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class OnnxBackboneConfig:
    # MobileNetV2 ONNX for OpenCV DNN
    name: str = "mobilenetv2-7"
    input_size: int = 224


def _model_url(name: str) -> str:
    # Use a stable HF snapshot URL (works without auth).
    if name == "mobilenetv2-7":
        return "https://huggingface.co/webml/models/resolve/fed237a8b98322b47b622a78094212ed710f5a1c/mobilenetv2-7.onnx"
    raise ValueError(f"Unknown ONNX backbone: {name}")


def ensure_onnx_model(*, cfg: OnnxBackboneConfig, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / f"{cfg.name}.onnx"
    if dst.exists() and dst.stat().st_size > 1024 * 1024:
        return dst

    import requests

    url = _model_url(cfg.name)
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    tmp = dst.with_suffix(".tmp")
    with tmp.open("wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 256):
            if chunk:
                f.write(chunk)
    tmp.replace(dst)
    return dst


def extract_onnx_features_from_rgb_list(
    rgbs: Iterable[np.ndarray],
    *,
    cfg: OnnxBackboneConfig,
    model_path: Path,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Feature extraction via OpenCV DNN on a pretrained ONNX backbone.

    Note: We use the logits output as a compact, strong descriptor.
    Output: [N, 1000] float32
    """
    import cv2

    net = cv2.dnn.readNetFromONNX(str(model_path))

    xs = list(rgbs)
    if not xs:
        return np.zeros((0, 1000), dtype=np.float32)

    size = int(cfg.input_size)
    feats: list[np.ndarray] = []

    # MobileNetV2 expects NCHW, RGB, normalized by ImageNet mean/std.
    def prep(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        x = cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)
        x = x / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean[None, None, :]) / std[None, None, :]
        x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
        return x

    for i in range(0, len(xs), int(batch_size)):
        batch = xs[i : i + int(batch_size)]
        blob = np.stack([prep(x) for x in batch], axis=0)
        net.setInput(blob)
        out = net.forward()
        out = out.reshape(out.shape[0], -1).astype(np.float32)
        feats.append(out)

    return np.concatenate(feats, axis=0)

