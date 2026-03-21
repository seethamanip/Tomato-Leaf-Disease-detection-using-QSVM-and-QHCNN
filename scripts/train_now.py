from __future__ import annotations
import sys
import traceback
from src.train_qsvm import main
if __name__ == "__main__":
    try:
        main()
        print("TRAIN_OK")
    except Exception as e:
        print("TRAIN_FAIL")
        traceback.print_exc()
        sys.exit(1)
