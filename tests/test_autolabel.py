import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from dmotion.autolabel import _draft_boxes, auto_label
from dmotion.config import AppConfig
from dmotion.dataset import Dataset
from dmotion.detector import Detection


def test_drafts_clip_valid_boxes_and_discard_empty_or_identical_rectangles():
    detections = [
        Detection((-10, 5, 110, 90), 0.8, "banknotes"),
        Detection((0, 5, 100, 80), 0.7, "cash"),
        Detection((110, 20, 120, 30), 0.6, "cash"),
    ]
    boxes, scores, labels = _draft_boxes(detections, 100, 80)
    assert boxes == [[0, 5, 100, 80]]
    assert scores == [0.8]
    assert labels == ["banknotes"]


def test_same_cash_with_different_prompt_boxes_is_suppressed_but_separate_cash_remains():
    detections = [
        Detection((2, 2, 42, 42), 0.9, "banknotes"),
        Detection((3, 2, 44, 44), 0.8, "dollar bills"),
        Detection((60, 10, 95, 60), 0.7, "cash money"),
    ]
    boxes, scores, labels = _draft_boxes(detections, 100, 80)
    assert boxes == [[2, 2, 42, 42], [60, 10, 95, 60]]
    assert scores == [0.9, 0.7]
    assert labels == ["banknotes", "cash money"]


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_model_scores_are_not_saved_as_suggestions(score):
    with pytest.raises(ValueError, match="confidence"):
        _draft_boxes([Detection((0, 0, 50, 50), score, "cash")], 100, 80)


def test_batch_labels_only_pending_records_and_keeps_empty_results_pending(tmp_path, monkeypatch):
    dataset = Dataset(tmp_path / "dataset")
    records = []
    for index in range(3):
        source = tmp_path / f"frame-{index}.jpg"
        source.write_bytes(f"image {index}".encode())
        records.append(dataset.add_image(source, group="one-video", width=100, height=80))
    dataset.review(records[0]["id"], status="negative", boxes=[])
    original = dataset.get_record(records[0]["id"])
    checkpoint = tmp_path / "models" / "yolov8s-worldv2.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"test detector weights")
    results = iter([[Detection((-2, 10, 90, 70), 0.75, "paper money")], []])
    reads = []

    class Detector:
        device = "cpu"
        model = SimpleNamespace()

        def __init__(self, config):
            pass

        def predict(self, frame):
            return next(results)

    def read(path):
        reads.append(path)
        return SimpleNamespace(shape=(80, 100, 3))

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(imread=read))
    monkeypatch.setattr("dmotion.autolabel.MoneyDetector", Detector)
    monkeypatch.setattr("dmotion.autolabel._contact_sheets", lambda *args: [])
    report = auto_label(AppConfig(root=tmp_path), dataset.directory, output=tmp_path / "output")
    summary = json.loads(report.read_text())
    assert summary["frames"] == 2
    assert summary["frames_with_boxes"] == 1
    assert summary["boxes"] == 1
    assert len(reads) == 2
    assert dataset.get_record(records[0]["id"]) == original
    cash = dataset.get_record(records[1]["id"])
    empty = dataset.get_record(records[2]["id"])
    assert cash["status"] == empty["status"] == "unreviewed"
    assert cash["boxes"] == empty["boxes"] == []
    assert cash["suggestion"]["boxes"] == [[0, 10, 90, 70]]
    assert empty["suggestion"]["boxes"] == []


def test_source_tampering_stops_before_creating_a_suggestion(tmp_path, monkeypatch):
    dataset = Dataset(tmp_path / "dataset")
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"original image")
    record = dataset.add_image(image, group="video", width=100, height=80)
    dataset.image_path(record).write_bytes(b"changed image")
    model = tmp_path / "model.pt"
    model.write_bytes(b"test weights")
    config = AppConfig(root=tmp_path)
    config = replace(config, detector=replace(config.detector, model=str(model)))
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace())
    monkeypatch.setattr(
        "dmotion.autolabel.MoneyDetector",
        lambda config: SimpleNamespace(device="cpu", model=SimpleNamespace()),
    )
    before = dataset.manifest_path.read_bytes()
    with pytest.raises(ValueError, match="changed since import"):
        auto_label(config, dataset.directory, output=tmp_path / "output")
    assert dataset.manifest_path.read_bytes() == before


@pytest.mark.parametrize("variant", ["tiny", "base"])
def test_grounding_variants_pair_model_id_with_immutable_revision(tmp_path, monkeypatch, variant):
    import sys
    from types import SimpleNamespace

    from dmotion.autolabel import auto_label
    from dmotion.config import AppConfig
    from dmotion.dataset import Dataset
    from dmotion.teachers import GROUNDING_MODELS

    path = tmp_path / "frame.jpg"
    path.write_bytes(b"image")
    dataset = Dataset(tmp_path / "dataset")
    dataset.add_image(path, group="recording", width=100, height=100)
    parameters = {}

    class Teacher:
        device = "cpu"
        model = SimpleNamespace(config=SimpleNamespace())

        def __init__(self, root, **kwargs):
            parameters.update(kwargs)
            self.revision = kwargs["revision"]

        def predict(self, frame):
            return []

    monkeypatch.setitem(
        sys.modules, "dmotion.grounding", SimpleNamespace(GroundingMoneyDetector=Teacher)
    )
    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(imread=lambda path: SimpleNamespace(shape=(100, 100, 3))),
    )
    monkeypatch.setattr("dmotion.autolabel._contact_sheets", lambda *args: [])
    report = auto_label(
        AppConfig(root=tmp_path), dataset.directory, engine="grounding", grounding_model=variant
    )
    model, revision = GROUNDING_MODELS[variant]
    assert parameters["model_id"] == model and parameters["revision"] == revision
    assert len(revision) == 40
    assert json.loads(report.read_text())["model_revision"] == revision
    assert dataset.summary()["unreviewed"] == 1
