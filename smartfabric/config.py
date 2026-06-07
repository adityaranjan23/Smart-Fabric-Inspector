from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ASSET_DIR = PROJECT_ROOT / "assets"
SAMPLE_DIR = ASSET_DIR / "samples"
DATASET_DIR = PROJECT_ROOT / "data" / "fabric"

DEFAULT_MODEL_PATH = MODEL_DIR / "best.pt"
DEFAULT_DATASET_ZIP = Path(r"E:\FAb.yolov8.zip")

DEFAULT_CLASS_NAMES = {
    0: "defect",
    1: "hole",
    2: "label",
    3: "paint",
    4: "staple",
}

SEVERITY_RULES = {
    "hole": "High",
    "cut": "High",
    "crack": "High",
    "thread break": "High",
    "oil spot": "Medium",
    "stain": "Medium",
    "paint": "Medium",
    "knot": "Medium",
    "staple": "Medium",
    "color variation": "Low",
    "label": "Low",
    "defect": "Medium",
}
