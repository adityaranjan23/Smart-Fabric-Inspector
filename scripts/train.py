from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smartfabric.config import DATASET_DIR, MODEL_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SmartFabric Inspector with YOLO.")
    parser.add_argument("--data", type=Path, default=DATASET_DIR / "data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO checkpoint, e.g. yolov8n.pt, yolo11n.pt, yolov8n-seg.pt")
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", type=Path, default=Path("runs") / "smartfabric")
    parser.add_argument("--name", default="fabric_yolo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}. Run scripts/prepare_dataset.py first.")

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(args.model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(args.project),
        name=args.name,
        pretrained=True,
        patience=20,
        cos_lr=True,
        close_mosaic=10,
        plots=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if best.exists():
        shutil.copy2(best, MODEL_DIR / "best.pt")
        print(f"Copied best checkpoint to {MODEL_DIR / 'best.pt'}")
    print(f"Training artifacts: {results.save_dir}")


if __name__ == "__main__":
    main()
