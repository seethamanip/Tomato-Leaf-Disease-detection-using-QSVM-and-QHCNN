from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TLConfig:
    backbone: str = "mobilenet_v3_small"
    batch_size: int = 64
    random_state: int = 42


def _get_backbone(name: str):
    import torch
    import torchvision.models as models

    if name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        m.classifier = torch.nn.Identity()
        out_dim = 576
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        return m, out_dim, weights
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        m.fc = torch.nn.Identity()
        out_dim = 512
        weights = models.ResNet18_Weights.DEFAULT
        return m, out_dim, weights

    raise ValueError(f"Unknown backbone: {name}")


def extract_tl_features_from_rgb_list(
    rgbs: Iterable[np.ndarray],
    *,
    cfg: TLConfig,
) -> np.ndarray:
    """
    Extract deep features using a pretrained CNN backbone.
    Returns a 2D array: [N, D]
    """
    import torch
    from torch.utils.data import DataLoader, Dataset
    import torchvision.transforms as T

    model, out_dim, weights = _get_backbone(cfg.backbone)
    model.eval()

    preprocess = weights.transforms()

    class _DS(Dataset):
        def __init__(self, xs):
            self.xs = list(xs)

        def __len__(self):
            return len(self.xs)

        def __getitem__(self, idx):
            x = self.xs[idx]
            # x is RGB uint8 numpy
            img = torch.from_numpy(x).permute(2, 0, 1).float() / 255.0
            img = preprocess(img)
            return img

    ds = _DS(rgbs)
    dl = DataLoader(ds, batch_size=int(cfg.batch_size), shuffle=False, num_workers=0)

    feats = []
    with torch.no_grad():
        for batch in dl:
            y = model(batch)
            y = y.reshape(y.shape[0], -1).cpu().numpy().astype(np.float32)
            feats.append(y)

    return np.concatenate(feats, axis=0) if feats else np.zeros((0, out_dim), dtype=np.float32)

