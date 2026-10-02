"""Inference progress and alert confirmation, independent of preview refresh rate."""

from dataclasses import dataclass, field

from dmotion.detector import Detection
from dmotion.trigger import DetectionTrigger


@dataclass
class InferenceMonitor:
    max_result_age_seconds: float
    trigger: DetectionTrigger
    completed: int = 0
    discarded: int = 0
    inference_ms: float = 0.0
    captured_at: float = float("-inf")
    last_match: Detection | None = None
    last_result_stale: bool = False
    error: str | None = None
    detections: list[Detection] = field(default_factory=list)

    def complete(self, detections: list[Detection], captured_at: float, finished_at: float) -> bool:
        self.completed += 1
        self.captured_at = captured_at
        self.inference_ms = (finished_at - captured_at) * 1000
        self.last_result_stale = finished_at - captured_at > self.max_result_age_seconds
        if self.last_result_stale:
            self.discarded += 1
            self.detections = []
            return False
        self.detections = detections
        if detections:
            self.last_match = max(detections, key=lambda detection: detection.confidence)
        # Only actual fresh inference results are observations for the alert gate.
        return self.trigger.update(bool(detections), finished_at)

    def visible(self, now: float) -> list[Detection]:
        """Expiry hides old boxes but never invents a negative detection sample."""
        if now - self.captured_at > self.max_result_age_seconds:
            return []
        return self.detections

    def headline(self, now: float, *, checking: bool, confidence: float) -> str:
        if self.error:
            return "AI stopped: see error below"
        if not self.completed:
            return "AI warming up... waiting for the first result"
        if self.last_result_stale:
            return "AI too slow: last result discarded"
        if checking and self.last_match:
            return f"AI CHECK PASSED: recognized {self.last_match.label}"
        if self.visible(now):
            return "OBJECT DETECTED" if checking else "MONEY DETECTED"
        if now - self.captured_at > self.max_result_age_seconds:
            return "AI running... waiting for a fresh result"
        return f"AI running: no match above {confidence:.2f}"
