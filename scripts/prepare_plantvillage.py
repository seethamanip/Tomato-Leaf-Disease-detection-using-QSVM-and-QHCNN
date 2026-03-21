from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.config import FOLDER_TO_CLASS, Paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Copy PlantVillage tomato subset into data/tomato/")
    ap.add_argument(
        "--src",
        required=True,
        help="Path to PlantVillage tomato directory that contains class folders like Tomato___Early_blight",
    )
    ap.add_argument(
        "--dst",
        default=str(Paths().data_root),
        help="Destination dataset root (default: data/tomato)",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination class folders if they exist",
    )
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    dst.mkdir(parents=True, exist_ok=True)

    class_folders = [name for name in FOLDER_TO_CLASS.keys() if name.startswith("Tomato___")]
    found_any = False

    for folder_name in sorted(set(class_folders)):
        src_dir = src / folder_name
        if not src_dir.exists():
            continue
        found_any = True
        dst_dir = dst / folder_name
        if dst_dir.exists() and args.overwrite:
            shutil.rmtree(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)

        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            for p in src_dir.glob(ext):
                shutil.copy2(p, dst_dir / p.name)

        print(f"Copied: {src_dir} -> {dst_dir}")

    if not found_any:
        raise SystemExit(
            f"No PlantVillage tomato class folders found in {src}. "
            f"Expected folders like Tomato___Early_blight."
        )

    print(f"Done. Dataset ready under: {dst}")


if __name__ == "__main__":
    main()

