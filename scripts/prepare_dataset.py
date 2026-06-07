from __future__ import annotations

import argparse
import ast
import random
import shutil
import sys
import zipfile
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smartfabric.config import DATASET_DIR, DEFAULT_DATASET_ZIP, SAMPLE_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract the Roboflow YOLO dataset and create train/valid/test splits.")
    parser.add_argument("--zip", type=Path, default=DEFAULT_DATASET_ZIP, help="Path to FAb.yolov8.zip")
    parser.add_argument("--out", type=Path, default=DATASET_DIR, help="Dataset output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    return parser.parse_args()


def copy_pair(image_path: Path, source_root: Path, target_root: Path, split: str) -> None:
    label_path = source_root / "train" / "labels" / f"{image_path.stem}.txt"
    image_target = target_root / split / "images" / image_path.name
    label_target = target_root / split / "labels" / label_path.name
    image_target.parent.mkdir(parents=True, exist_ok=True)
    label_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, image_target)
    if label_path.exists():
        shutil.copy2(label_path, label_target)
    else:
        label_target.write_text("", encoding="utf-8")


def rewrite_dataset_yaml_without_pyyaml(source_yaml: Path, dataset_dir: Path) -> str:
    nc = 0
    names: list[str] = []
    for line in source_yaml.read_text(encoding="utf-8").splitlines():
        if line.startswith("nc:"):
            nc = int(line.split(":", 1)[1].strip())
        elif line.startswith("names:"):
            names = ast.literal_eval(line.split(":", 1)[1].strip())

    if not names:
        raise RuntimeError("Could not read class names from source data.yaml. Install pyyaml and retry.")
    if not nc:
        nc = len(names)

    return (
        f"path: {dataset_dir.resolve()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        f"nc: {nc}\n"
        f"names: {names}\n"
    )


def main() -> None:
    args = parse_args()
    if not args.zip.exists():
        raise FileNotFoundError(f"Dataset zip not found: {args.zip}")

    raw_dir = args.out / "_raw"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {args.zip} -> {raw_dir}")
    with zipfile.ZipFile(args.zip) as archive:
        archive.extractall(raw_dir)

    image_dir = raw_dir / "train" / "images"
    images = sorted([*image_dir.glob("*.jpg"), *image_dir.glob("*.png"), *image_dir.glob("*.jpeg")])
    if not images:
        raise RuntimeError(f"No images found under {image_dir}")

    random.seed(args.seed)
    random.shuffle(images)
    test_count = int(len(images) * args.test_ratio)
    valid_count = int(len(images) * args.valid_ratio)
    split_map = {
        "test": images[:test_count],
        "valid": images[test_count : test_count + valid_count],
        "train": images[test_count + valid_count :],
    }

    for split in ("train", "valid", "test"):
        split_root = args.out / split
        if split_root.exists():
            shutil.rmtree(split_root)
        for image in split_map[split]:
            copy_pair(image, raw_dir, args.out, split)

    source_yaml = raw_dir / "data.yaml"
    if yaml is not None:
        data = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
        data["path"] = str(args.out.resolve())
        data["train"] = "train/images"
        data["val"] = "valid/images"
        data["test"] = "test/images"
        yaml_text = yaml.safe_dump(data, sort_keys=False)
    else:
        yaml_text = rewrite_dataset_yaml_without_pyyaml(source_yaml, args.out)
    (args.out / "data.yaml").write_text(yaml_text, encoding="utf-8")

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for sample in images[:6]:
        shutil.copy2(sample, SAMPLE_DIR / sample.name)

    print("Prepared dataset:")
    for split, split_images in split_map.items():
        print(f"  {split}: {len(split_images)} images")
    print(f"  data: {args.out / 'data.yaml'}")
    print(f"  samples: {SAMPLE_DIR}")


if __name__ == "__main__":
    main()
