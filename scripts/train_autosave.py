from __future__ import annotations
from src.config import Paths
from src.infer import load_bundle
if __name__ == "__main__":
    load_bundle(Paths())
    print("MODEL_OK")
