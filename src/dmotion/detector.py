"""Prompt-based and trained YOLO adapters, loaded only for actual inference."""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dmotion.config import AppConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    box: tuple[int, int, int, int]
    confidence: float
    label: str


def mirror_detections(detections: list[Detection], width: int) -> list[Detection]:
    """Map original-image boxes onto a mirrored preview without mirroring model input."""
    return [
        Detection(
            (width - item.box[2], item.box[1], width - item.box[0], item.box[3]),
            item.confidence,
            item.label,
        )
        for item in detections
    ]


def prepare_environment(root: Path) -> None:
    """Keep downloaded files and library settings in ignored project folders."""
    cache = root / ".cache"
    # Ultralytics checks write access before creating its nested config directory.
    (cache / "ultralytics" / "Ultralytics").mkdir(parents=True, exist_ok=True)
    (cache / "matplotlib").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(cache / "ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")


def select_device(requested: str, torch) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class MoneyDetector:
    def __init__(self, config: AppConfig):
        prepare_environment(config.root)
        import torch
        from ultralytics import YOLO, YOLOWorld, settings

        self.config = config.detector
        self.training_note = ""
        self.device = select_device(self.config.device, torch)
        model_path = config.resolve(self.config.model)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        # Keep the library's default weights directory stable for the CLIP cache.
        settings.update({"sync": False})
        logger.info("Loading %s on %s", model_path.name, self.device)
        if self.config.backend == "trained":
            if not model_path.is_file():
                raise FileNotFoundError(
                    "No trained model yet. Review your photos and run make train."
                )
            self.model = YOLO(str(model_path))
            if set(self.model.names.values()) != {"money_spread"}:
                raise ValueError("The trained model must contain only the money_spread class")
            report = model_path.with_suffix(".json")
            if report.is_file():
                try:
                    info = json.loads(report.read_text())
                    count = sum(split["images"] for split in info["splits"].values())
                    if count < 100:
                        self.training_note = f"STARTER MODEL: {count} photos. Add webcam examples."
                        logger.warning(self.training_note)
                except (ValueError, TypeError, KeyError, AttributeError, OSError):
                    logger.warning("Could not read the training report: %s", report)
        else:
            self.model = YOLOWorld(str(model_path))
            # Encode vocabulary once on CPU, then move to the runtime device.
            self.model.set_classes(list(self.config.prompts))
        self.model.to(self.device)

    def predict(self, frame) -> list[Detection]:
        result = self.model.predict(
            frame,
            conf=self.config.confidence,
            imgsz=self.config.image_size,
            device=self.device,
            verbose=False,
        )[0]
        if result.boxes is None:
            return []
        return [
            Detection(tuple(round(v) for v in box), float(score), result.names[int(class_id)])
            for box, score, class_id in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
                strict=True,
            )
        ]
