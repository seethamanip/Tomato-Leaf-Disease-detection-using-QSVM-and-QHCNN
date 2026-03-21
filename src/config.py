from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    project_root: Path = Path(__file__).resolve().parents[1]
    data_root: Path = project_root / "data" / "tomato"
    artifacts_dir: Path = project_root / "artifacts"
    model_path: Path = artifacts_dir / "model.joblib"
    metrics_path: Path = artifacts_dir / "metrics.json"
    cm_path: Path = artifacts_dir / "confusion_matrix.png"
    qhcnn_model_path: Path = artifacts_dir / "qhcnn_model.joblib"
    qhcnn_metrics_path: Path = artifacts_dir / "qhcnn_metrics.json"


# Canonical classes for outputs (model trains on what it finds in your dataset)
CANONICAL_CLASSES: list[str] = [
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Septoria Leaf Spot",
    "Tomato Bacterial Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Healthy Leaf",
]


# Map dataset folder names -> canonical class names
# Adjust these if your dataset uses different folder names.
FOLDER_TO_CLASS: dict[str, str] = {
    # --- Generic/simple folder names (if you curated a custom subset) ---
    "Tomato_Early_Blight": "Tomato Early Blight",
    "Tomato_Late_Blight": "Tomato Late Blight",
    "Tomato_Leaf_Mold": "Tomato Leaf Mold",
    "Tomato_Septoria_Leaf_Spot": "Tomato Septoria Leaf Spot",
    "Tomato_Bacterial_Spot": "Tomato Bacterial Spot",
    "Tomato_Yellow_Leaf_Curl_Virus": "Tomato Yellow Leaf Curl Virus",
    "Healthy_Leaf": "Healthy Leaf",

    # --- PlantVillage variant folder names found in this workspace ---
    "Tomato_Bacterial_spot": "Tomato Bacterial Spot",
    "Tomato_Early_blight": "Tomato Early Blight",
    "Tomato_Late_blight": "Tomato Late Blight",
    "Tomato_Leaf_Mold": "Tomato Leaf Mold",
    "Tomato_Septoria_leaf_spot": "Tomato Septoria Leaf Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomato Yellow Leaf Curl Virus",
    "Tomato_healthy": "Healthy Leaf",

    # --- PlantVillage tomato subset canonical folder names ---
    # (Non-target PlantVillage classes like Target Spot / Spider mites / Mosaic virus are intentionally not mapped.)
    "Tomato___Bacterial_spot": "Tomato Bacterial Spot",
    "Tomato___Early_blight": "Tomato Early Blight",
    "Tomato___Late_blight": "Tomato Late Blight",
    "Tomato___Leaf_Mold": "Tomato Leaf Mold",
    "Tomato___Septoria_leaf_spot": "Tomato Septoria Leaf Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato Yellow Leaf Curl Virus",
    "Tomato___healthy": "Healthy Leaf",
}


@dataclass(frozen=True)
class ImageConfig:
    size: tuple[int, int] = (128, 128)
    max_upload_mb: int = 8


@dataclass(frozen=True)
class FeatureConfig:
    # HSV histogram bins (per channel)
    hsv_bins: int = 32
    # Canny thresholds
    canny_low: int = 50
    canny_high: int = 150
    # GLCM configuration
    glcm_distances: tuple[int, ...] = (1, 2, 4)
    glcm_angles: tuple[float, ...] = (0.0, 0.78539816339, 1.57079632679, 2.35619449019)  # 0, pi/4, pi/2, 3pi/4
    glcm_levels: int = 64  # finer quantization for richer texture stats


@dataclass(frozen=True)
class TrainConfig:
    # Use 20% of the data for testing so 80% is used for training.
    test_size: float = 0.20
    random_state: int = 42
    pca_components: int = 32  # keep more variance for higher accuracy
    feature_map_reps: int = 2
    # Set to 0 to disable per-class capping and use the full dataset.
    max_samples_per_class: int = 0
