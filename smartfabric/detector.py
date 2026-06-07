from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any

import cv2
import numpy as np
import torch

from smartfabric.config import DEFAULT_CLASS_NAMES, SEVERITY_RULES

cv2.setUseOptimized(True)
torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


@dataclass
class Detection:
    frame_id: int
    class_id: int
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float
    area: float
    severity: str


class SmartFabricDetector:
    """Thin production wrapper around Ultralytics YOLO for fabric inspection."""

    def __init__(self, model_path: str | Path | None = None, task: str = "detect") -> None:
        self.model_path = Path(model_path).expanduser() if model_path else None
        self.task = task
        self.model: Any | None = None
        self.names: dict[int, str] = DEFAULT_CLASS_NAMES.copy()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if self.model_path and self.model_path.exists():
            self._load_model()
            if getattr(self.model, "names", None):
                self.names = {int(k): str(v) for k, v in self.model.names.items()}

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def _load_model(self) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(self.model_path), task=self.task)
        self._disable_repeated_fuse()

    def _disable_repeated_fuse(self) -> None:
        """Avoid an Ultralytics/PyTorch edge case where an already-fused model is fused again."""
        if self.model is None or not hasattr(self.model, "model"):
            return

        def no_op_fuse(model_self, verbose: bool = True):
            return model_self

        self.model.model.fuse = MethodType(no_op_fuse, self.model.model)

    def predict(
        self,
        image: np.ndarray,
        confidence: float = 0.35,
        iou: float = 0.45,
        imgsz: int = 320,
        max_det: int = 50,
    ) -> tuple[np.ndarray, list[Detection]]:
        if self.model is None:
            return self._demo_predict(image, confidence)

        try:
            results = self._predict_ultralytics(image, confidence, iou, imgsz, max_det)
        except AttributeError as exc:
            if "object has no attribute 'bn'" not in str(exc) or self.model_path is None:
                raise
            self._load_model()
            results = self._predict_ultralytics(image, confidence, iou, imgsz, max_det)
        result = results[0]
        annotated = result.plot()
        detections: list[Detection] = []

        if result.boxes is not None:
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            confs = result.boxes.conf.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            for box, conf, class_id in zip(xyxy, confs, classes):
                x1, y1, x2, y2 = [float(v) for v in box]
                label = self.names.get(class_id, f"class_{class_id}")
                detections.append(
                    Detection(
                        frame_id=0,
                        class_id=class_id,
                        label=label,
                        confidence=float(conf),
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        width=x2 - x1,
                        height=y2 - y1,
                        area=(x2 - x1) * (y2 - y1),
                        severity=severity_for(label, (x2 - x1) * (y2 - y1), image.shape[:2]),
                    )
                )

        return annotated, detections

    def _predict_ultralytics(
        self,
        image: np.ndarray,
        confidence: float,
        iou: float,
        imgsz: int,
        max_det: int,
    ):
        with torch.inference_mode():
            return self.model.predict(
                source=image,
                conf=confidence,
                iou=iou,
                imgsz=imgsz,
                max_det=max_det,
                device=self.device,
                verbose=False,
                half=self.device.startswith("cuda"),
                retina_masks=self.task == "segment",
            )

    def _demo_predict(self, image: np.ndarray, confidence: float) -> tuple[np.ndarray, list[Detection]]:
        """Deterministic visual demo for environments without a trained model."""
        h, w = image.shape[:2]
        boxes = [
            (0.18 * w, 0.20 * h, 0.36 * w, 0.38 * h, "hole", 0.91),
            (0.58 * w, 0.52 * h, 0.82 * w, 0.68 * h, "paint", 0.84),
        ]
        annotated = image.copy()
        detections: list[Detection] = []
        for idx, (x1, y1, x2, y2, label, conf) in enumerate(boxes):
            if conf < confidence:
                continue
            color = (36, 180, 87) if label != "hole" else (42, 69, 240)
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            caption = f"DEMO {label} {conf:.2f}"
            cv2.rectangle(annotated, (int(x1), max(0, int(y1) - 24)), (int(x1) + 165, int(y1)), color, -1)
            cv2.putText(annotated, caption, (int(x1) + 6, int(y1) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            detections.append(
                Detection(
                    frame_id=0,
                    class_id=idx,
                    label=label,
                    confidence=conf,
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    width=float(x2 - x1),
                    height=float(y2 - y1),
                    area=float((x2 - x1) * (y2 - y1)),
                    severity=severity_for(label, (x2 - x1) * (y2 - y1), image.shape[:2]),
                )
            )
        return annotated, detections


def severity_for(label: str, area: float, image_shape: tuple[int, int]) -> str:
    base = SEVERITY_RULES.get(label.lower(), "Medium")
    image_area = max(float(image_shape[0] * image_shape[1]), 1.0)
    area_ratio = area / image_area
    if area_ratio >= 0.08:
        return "High"
    if area_ratio <= 0.01 and base == "Medium":
        return "Low"
    return base
