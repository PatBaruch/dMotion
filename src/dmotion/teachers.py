"""Optional offline labeling teachers; outputs are drafts, never reviewed labels."""

import hashlib
import json
import math
from pathlib import Path

from dmotion.dataset import _validate_boxes
from dmotion.detector import Detection, prepare_environment, select_device

GROUNDING_MODELS = {
    "tiny": "IDEA-Research/grounding-dino-tiny",
    "base": "IDEA-Research/grounding-dino-base",
}


class ReferenceTeacher:
    """YOLOE seeded with one explicitly boxed reference image, independent of target frames."""

    def __init__(
        self,
        root: Path,
        model: Path,
        reference: Path,
        *,
        confidence: float,
        image_size: int,
        device: str,
    ):
        prepare_environment(root)
        import cv2
        import numpy as np
        import torch
        from ultralytics import YOLOE
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

        if not model.is_file():
            raise ValueError(f"Download YOLOE weights before labeling: {model}")
        data = json.loads(reference.read_text())
        if data.get("version") != 1:
            raise ValueError("Expected a version 1 reference file")
        source = (reference.parent / data["image"]).resolve()
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != data.get("sha256"):
            raise ValueError("Reference image changed; create the reference again")
        frame = cv2.imread(str(source))
        if frame is None:
            raise ValueError("Cannot decode reference image")
        box = data["box"]
        if (
            not isinstance(box, list)
            or len(box) != 4
            or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in box
            )
        ):
            raise ValueError("Reference box must contain four finite coordinates")
        if not (0 <= box[0] < box[2] <= frame.shape[1] and 0 <= box[1] < box[3] <= frame.shape[0]):
            raise ValueError("Reference box must fit the reference image")
        _validate_boxes("positive", [[round(v) for v in box]], frame.shape[1], frame.shape[0])
        self.device = select_device(device, torch)
        self.confidence, self.image_size = confidence, image_size
        self.reference_provenance = {
            "file": str(reference),
            "image_sha256": digest,
            "box": data["box"],
        }
        self.model = YOLOE(str(model))
        self.model.predict(
            frame,
            refer_image=frame,
            visual_prompts={"bboxes": np.array([data["box"]]), "cls": np.array([0])},
            predictor=YOLOEVPSegPredictor,
            conf=confidence,
            imgsz=image_size,
            device=self.device,
            verbose=False,
        )

    def predict(self, frame):
        result = self.model.predict(
            frame, conf=self.confidence, imgsz=self.image_size, device=self.device, verbose=False
        )[0]
        if result.boxes is None:
            return []
        return [
            Detection(tuple(round(v) for v in box), float(score), "cash")
            for box, score in zip(
                result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), strict=True
            )
        ]
