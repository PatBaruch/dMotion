import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from dmotion.teachers import ReferenceTeacher


def dependencies(monkeypatch):
    calls = []
    frame = SimpleNamespace(shape=(720, 1280, 3))

    class Model:
        def __init__(self, path):
            self.model = SimpleNamespace()

        def predict(self, image, **kwargs):
            calls.append((image, kwargs))
            return [SimpleNamespace(boxes=None)]

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(imread=lambda path: frame))
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace(array=lambda value: value))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLOE=Model))
    monkeypatch.setitem(
        sys.modules,
        "ultralytics.models.yolo.yoloe",
        SimpleNamespace(YOLOEVPSegPredictor="visual-prompt-predictor"),
    )
    monkeypatch.setattr("dmotion.teachers.select_device", lambda device, torch: "cpu")
    return frame, calls


def reference_file(tmp_path):
    source = tmp_path / "cash.jpg"
    source.write_bytes(b"reference image")
    data = {
        "version": 1,
        "image": "cash.jpg",
        "box": [265.0, 137.0, 832.0, 720.0],
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps(data))
    model = tmp_path / "yoloe.pt"
    model.write_bytes(b"weights")
    return source, reference, model, data


def test_yoloe_uses_explicit_cash_reference_then_reuses_it_on_target_frames(tmp_path, monkeypatch):
    frame, calls = dependencies(monkeypatch)
    _, reference, model, _ = reference_file(tmp_path)
    teacher = ReferenceTeacher(
        tmp_path, model, reference, confidence=0.25, image_size=640, device="cpu"
    )
    assert calls[0][1]["refer_image"] is frame
    assert calls[0][1]["visual_prompts"]["bboxes"] == [[265.0, 137.0, 832.0, 720.0]]
    target = object()
    assert teacher.predict(target) == []
    assert calls[1][0] is target and "refer_image" not in calls[1][1]
    assert teacher.reference_provenance["image_sha256"]


def test_reference_image_hash_changes_are_rejected_before_model_loading(tmp_path, monkeypatch):
    _, calls = dependencies(monkeypatch)
    source, reference, model, _ = reference_file(tmp_path)
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Reference image changed"):
        ReferenceTeacher(tmp_path, model, reference, confidence=0.25, image_size=640, device="cpu")
    assert calls == []


@pytest.mark.parametrize("box", [[0, 0, 0, 1], [0, 0, 1281, 720], [0, 0, float("nan"), 10]])
def test_invalid_reference_boxes_fail(tmp_path, monkeypatch, box):
    dependencies(monkeypatch)
    _, reference, model, data = reference_file(tmp_path)
    data["box"] = box
    reference.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Reference box"):
        ReferenceTeacher(tmp_path, model, reference, confidence=0.25, image_size=640, device="cpu")
