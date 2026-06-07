from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smartfabric.config import DATASET_DIR, DEFAULT_MODEL_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SmartFabric YOLO model.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data", type=Path, default=DATASET_DIR / "data.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    device = 0 if torch.cuda.is_available() else "cpu"
    metrics = YOLO(str(args.model)).val(data=str(args.data), split=args.split, imgsz=args.imgsz, device=device, plots=True)
    print(metrics)


if __name__ == "__main__":
    main()
