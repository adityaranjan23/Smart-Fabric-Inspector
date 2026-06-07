# SmartFabric Inspector

SmartFabric Inspector is a deployable Streamlit application for real-time fabric defect detection in textile quality-control workflows. It uses Ultralytics YOLO for detection or segmentation, OpenCV for image/video processing, and built-in report generation for CSV/PDF inspection summaries.

## Features

- Live webcam inspection with bounding boxes, labels, and confidence scores
- Image and video upload with annotated output
- YOLOv8 or YOLO11 checkpoint support, including optional segmentation checkpoints
- Confidence and IoU controls
- Speed presets for CPU-friendly inference: Fast, Balanced, and Quality
- Automatic CPU/GPU detection
- Defect reports with counts, classes, severity, charts, CSV, and PDF downloads
- Light/dark UI mode
- Demo mode when `models/best.pt` is not available
- Dataset preparation and training scripts for the attached Roboflow YOLOv8 export

## Dataset Used

This project uses the `FAb.yolov8.zip` fabric defect dataset exported from Roboflow in YOLOv8 format.

- Approximate archive size: 1.06 GB
- Total images: 531
- Annotation format: YOLOv8 object detection labels
- Classes: `defect`, `hole`, `label`, `paint`, `staple`
- Prepared full split:
  - Train: 399 images
  - Validation: 79 images
  - Test: 53 images
- Annotated positive-only split used for the final lightweight checkpoint:
  - Train: 79 images
  - Validation: 14 images
  - Test: 6 images

The positive-only split was used because most images in the original dataset were background/empty samples. This improved visible defect detection for the demo checkpoint included in `models/best.pt`.

## Key Metrics Achieved

The included `models/best.pt` checkpoint was trained with YOLOv8n on CPU using the annotated positive-only split.

- Model: YOLOv8n
- Epochs: 20
- Image size: 416
- Batch size: 4
- Device used for training: CPU
- Validation precision: 0.988
- Validation recall: 0.500
- Validation mAP@50: 0.506
- Validation mAP@50-95: 0.412
- Approximate CPU inference speed observed: 0.18 to 0.30 seconds per image, depending on inference size and UI rendering

Per-class validation highlights:

- `hole`: mAP@50 approximately 0.995
- `label`: mAP@50 approximately 0.995
- `paint`: mAP@50 approximately 0.036
- `defect`: limited detection performance due to few and broad annotations

These metrics should be treated as baseline results because the dataset is small and imbalanced. Accuracy can be improved with a larger balanced dataset, more annotated defect categories, longer training, and GPU-based tuning.

## Project Layout

```text
.
|-- app.py
|-- smartfabric/
|   |-- config.py
|   |-- detector.py
|   |-- reporting.py
|   `-- visuals.py
|-- scripts/
|   |-- prepare_dataset.py
|   |-- train.py
|   `-- evaluate.py
|-- assets/samples/
|-- models/
|-- outputs/
|-- requirements.txt
`-- Dockerfile
```

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run app.py
```

The app looks for a trained checkpoint at `models/best.pt`. If it is missing, the UI runs in demo mode with deterministic sample overlays.

## Prepare the Attached Dataset

The attached archive is expected at `E:\FAb.yolov8.zip`. It contains a Roboflow YOLOv8 export with train images/labels and a `data.yaml`.

```powershell
python scripts\prepare_dataset.py --zip E:\FAb.yolov8.zip
```

The script extracts the data into `data/fabric`, creates train/valid/test splits, rewrites `data/fabric/data.yaml`, and copies a few examples into `assets/samples` for demo mode.

## Train YOLO

Small, fast detection baseline:

```powershell
python scripts\train.py --model yolov8n.pt --epochs 75 --imgsz 640 --batch 16
```

YOLO11 baseline:

```powershell
python scripts\train.py --model yolo11n.pt --epochs 75 --imgsz 640 --batch 16
```

Segmentation checkpoint, if you later add segmentation labels:

```powershell
python scripts\train.py --model yolo11n-seg.pt --epochs 75 --imgsz 640 --batch 8
```

After training, the best checkpoint is copied to `models/best.pt` for the Streamlit app.

## Evaluate

```powershell
python scripts\evaluate.py --model models\best.pt --split val
python scripts\evaluate.py --model models\best.pt --split test
```

## Run With Docker

```powershell
docker build -t smartfabric-inspector .
docker run --rm -p 8501:8501 smartfabric-inspector
```

For GPU inference in Docker, use an NVIDIA-enabled runtime and install a CUDA-compatible PyTorch build in the image.

## Production Notes

- Keep trained weights outside source control when model files become large; for this repository, `models/best.pt` is included because it is small.
- Use the confidence slider conservatively during commissioning, then lock a validated threshold per production fabric type.
- Track false positives and false negatives by fabric style, lighting setup, and camera distance; add hard examples back into the training dataset.
- For true browser-native live video at scale, Streamlit can be extended with `streamlit-webrtc`; this project keeps dependencies lean and provides browser capture plus workstation OpenCV webcam mode.
