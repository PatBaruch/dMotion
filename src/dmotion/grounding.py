"""Optional text-guided Grounding DINO adapter for draft cash annotations."""

import math
from pathlib import Path

from dmotion.detector import Detection, select_device

DEFAULT_GROUNDING_MODEL = "IDEA-Research/grounding-dino-tiny"
DEFAULT_CASH_PROMPTS = ("banknotes", "dollar bills", "cash money")


class GroundingMoneyDetector:
    """Predict cash boxes in original OpenCV-frame coordinates on CPU by default.

    Requires Transformers 4.57.x, PyTorch and Pillow only when instantiated.
    The first initialization downloads model assets into the project cache.
    """

    def __init__(
        self,
        root: Path,
        *,
        confidence: float = 0.25,
        text_threshold: float = 0.2,
        prompts: tuple[str, ...] = DEFAULT_CASH_PROMPTS,
        image_size: int = 800,
        device: str = "cpu",
        model_id: str = DEFAULT_GROUNDING_MODEL,
    ):
        if not 0 <= confidence <= 1 or not 0 <= text_threshold <= 1:
            raise ValueError("Grounding DINO thresholds must be between 0 and 1")
        if image_size <= 0:
            raise ValueError("Grounding DINO image size must be positive")
        if not prompts or any(not prompt.strip() for prompt in prompts):
            raise ValueError("Grounding DINO needs at least one nonempty cash prompt")
        try:
            import torch
            from PIL import Image
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Grounding DINO needs the optional grounding dependencies "
                "(transformers>=4.57,<4.58, torch and Pillow)."
            ) from exc

        self.device = select_device(device, torch)
        self.model_id = model_id
        self.confidence = confidence
        self.text_threshold = text_threshold
        self.prompts = tuple(prompt.strip().lower().rstrip(".") for prompt in prompts)
        self.image_size = image_size
        self._torch = torch
        self._image = Image
        self.cache_dir = Path(root) / ".cache" / "huggingface" / "hub"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=str(self.cache_dir),
            use_fast=False,
        )
        # CPU inference uses the regular PyTorch implementation; no CUDA build is needed.
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id,
            cache_dir=str(self.cache_dir),
            use_safetensors=True,
            disable_custom_kernels=True,
        ).to(self.device)
        self.model.eval()

    def predict(self, frame) -> list[Detection]:
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError("Grounding DINO expects a three-channel OpenCV BGR frame")
        height, width = frame.shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError("Grounding DINO cannot process an empty frame")
        image = self._image.fromarray(frame[:, :, ::-1].copy())
        inputs = self.processor(
            images=image,
            text=[list(self.prompts)],
            return_tensors="pt",
            size={
                "shortest_edge": self.image_size,
                "longest_edge": round(self.image_size * 1333 / 800),
            },
        ).to(self.device)
        with self._torch.inference_mode():
            outputs = self.model(**inputs)
        # Transformers 4.57 calls the box confidence option `threshold`.
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.confidence,
            text_threshold=self.text_threshold,
            target_sizes=[(height, width)],
        )[0]
        detections = []
        for box, score, label in zip(
            result["boxes"].cpu().tolist(),
            result["scores"].cpu().tolist(),
            result["text_labels"],
            strict=True,
        ):
            label = str(label).strip()
            # Low box thresholds can retain proposals with no token above the
            # text threshold. Such proposals have no usable annotation label.
            if not label:
                continue
            if not math.isfinite(score) or not self.confidence <= score <= 1:
                continue
            if len(box) != 4 or any(not math.isfinite(value) for value in box):
                continue
            x1, y1, x2, y2 = (round(value) for value in box)
            clipped = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
            if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
                continue
            detections.append(Detection(clipped, float(score), label))
        return detections
