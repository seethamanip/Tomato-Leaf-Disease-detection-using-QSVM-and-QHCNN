from __future__ import annotations
from src.config import Paths
from src.infer import _train_on_the_fly  # type: ignore
if __name__ == "__main__":
    bundle = _train_on_the_fly(Paths())
    print("FORCE_AUTOTRAIN_OK", len(bundle.get("y_train", [])))
