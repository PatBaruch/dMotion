import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from dmotion.autolabel import _draft_boxes, auto_label
from dmotion.config import AppConfig
from dmotion.dataset import Dataset, _write_json
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
    checkpoint = tmp_path / "models" / "cash-yolo26m.pt"
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


def test_existing_pending_draft_is_preserved_without_loading_model(tmp_path, monkeypatch):
    dataset = Dataset(tmp_path / "dataset")
    image = tmp_path / "draft.jpg"
    image.write_bytes(b"image")
    record = dataset.add_image(image, group="session", width=100, height=80)
    dataset.suggest(
        record["id"],
        {
            "boxes": [],
            "scores": [],
            "labels": [],
            "model": "old-model",
            "created_at": "2026-10-06T00:00:00+00:00",
            "prompts": ["cash"],
        },
    )
    before = dataset.manifest_path.read_bytes()
    monkeypatch.setitem(sys.modules, "cv2", None)
    monkeypatch.setattr("dmotion.autolabel._contact_sheets", lambda *args: [])
    report = auto_label(AppConfig(root=tmp_path), dataset.directory)
    summary = json.loads(report.read_text())
    assert summary["frames"] == summary["preserved_frames"] == 1
    assert summary["new_frames"] == 0
    assert summary["models"] == [
        {"model": "old-model", "model_sha256": None, "model_revision": None}
    ]
    predictions = json.loads(report.with_name("predictions.json").read_text())
    assert predictions["records"][0]["suggestion"]["model"] == "old-model"
    assert dataset.manifest_path.read_bytes() == before


@pytest.mark.parametrize("stage", ["during_inference", "before_report"])
def test_interrupted_labeling_resumes_complete_audit_without_replacing_drafts(
    tmp_path, monkeypatch, stage
):
    dataset = Dataset(tmp_path / "dataset")
    for index in range(2):
        image = tmp_path / f"frame-{index}.jpg"
        image.write_bytes(f"image {index}".encode())
        dataset.add_image(image, group="session", width=100, height=80)
    checkpoint = tmp_path / "models/cash-yolo26m.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    calls = []
    state = {"fail": True}

    class Detector:
        device = "cpu"

        def __init__(self, config):
            calls.append("load")

        def predict(self, frame):
            calls.append("predict")
            if stage == "during_inference" and state["fail"] and calls.count("predict") == 2:
                raise RuntimeError("interrupted")
            return [Detection((10, 10, 90, 70), 0.8, "cash")]

    sheets = []

    def contact_sheets(dataset, records, output):
        if stage == "before_report" and state["fail"]:
            raise RuntimeError("interrupted")
        sheets.extend(record["id"] for record in records)
        return []

    monkeypatch.setitem(
        sys.modules, "cv2", SimpleNamespace(imread=lambda _: SimpleNamespace(shape=(80, 100, 3)))
    )
    monkeypatch.setattr("dmotion.autolabel.MoneyDetector", Detector)
    monkeypatch.setattr("dmotion.autolabel._contact_sheets", contact_sheets)
    config = AppConfig(root=tmp_path)
    with pytest.raises(RuntimeError, match="interrupted"):
        auto_label(config, dataset.directory)
    saved = {
        record["id"]: record["suggestion"] for record in dataset.records() if "suggestion" in record
    }
    state["fail"] = False
    if stage == "before_report":
        # Regenerate artifacts after every draft is saved, even with no model available.
        checkpoint.unlink()
        monkeypatch.setitem(sys.modules, "cv2", None)
    report = auto_label(config, dataset.directory)
    info = json.loads(report.read_text())
    predictions = json.loads(report.with_name("predictions.json").read_text())["records"]
    assert info["frames"] == info["boxes"] == 2
    assert info["preserved_frames"] == len(saved)
    assert len(predictions) == len(sheets) == 2
    assert set(sheets) == {record["id"] for record in dataset.records()}
    assert all(record["status"] == "unreviewed" and not record["boxes"] for record in predictions)
    assert all(dataset.get_record(key)["suggestion"] == value for key, value in saved.items())
    if stage == "before_report":
        assert calls.count("load") == 1


@pytest.mark.parametrize("record_count", [13, 15])
@pytest.mark.parametrize(
    "failure_stage", [None, "render", "partial_render", "report", "after_commit"]
)
def test_saved_draft_rebuild_prunes_pages_only_after_committing_the_report(
    tmp_path, monkeypatch, failure_stage, record_count
):
    dataset = Dataset(tmp_path / "dataset")
    for index in range(record_count):
        image = tmp_path / f"frame-{index}.jpg"
        image.write_bytes(f"image {index}".encode())
        record = dataset.add_image(image, group="session", width=100, height=80)
        dataset.suggest(
            record["id"],
            {
                "boxes": [],
                "scores": [],
                "labels": [],
                "model": "saved-cash-model",
                "created_at": "2026-10-06T00:00:00+00:00",
                "prompts": ["cash"],
            },
        )

    state = {"rebuilding": False}

    def render(dataset, records, output):
        paths = []
        for page, offset in enumerate(range(0, len(records), 12), start=1):
            path = output / f"contact-sheet-{page:02}.jpg"
            path.write_text("\n".join(record["id"] for record in records[offset : offset + 12]))
            paths.append(str(path))
            if state["rebuilding"] and failure_stage == "partial_render" and page == 1:
                raise RuntimeError("partial render interrupted")
        return paths

    monkeypatch.setitem(sys.modules, "cv2", None)
    monkeypatch.setattr("dmotion.autolabel._contact_sheets", render)
    config = AppConfig(root=tmp_path)
    report = auto_label(config, dataset.directory)
    output = report.parent
    old_report = report.read_bytes()
    original_sheets = {
        Path(path): Path(path).read_bytes() for path in json.loads(old_report)["contact_sheets"]
    }
    obsolete = next(path for path in original_sheets if path.name == "contact-sheet-02.jpg")
    assert obsolete.is_file()
    legacy = output / "contact-sheet-99.jpg"
    legacy.write_bytes(b"legacy generated page")
    notes = output / "contact-sheet-notes.jpg"
    notes.write_bytes(b"user notes")
    generation_notes = obsolete.parent / "review-notes.txt"
    generation_notes.write_bytes(b"generation notes")
    reviewed = dataset.records()[:2]
    for record in reviewed:
        dataset.review(record["id"], status="negative", boxes=[])
    manifest = dataset.manifest_path.read_bytes()
    state["rebuilding"] = True
    if failure_stage == "render":

        def fail(*args):
            raise RuntimeError("render interrupted")

        monkeypatch.setattr("dmotion.autolabel._contact_sheets", fail)
    elif failure_stage in {"report", "after_commit"}:

        def write(path, value):
            if path.name == "report.json":
                if failure_stage == "after_commit":
                    _write_json(path, value)
                raise RuntimeError("report interrupted")
            _write_json(path, value)

        monkeypatch.setattr("dmotion.autolabel._write_json", write)
    if failure_stage:
        with pytest.raises(RuntimeError, match="interrupted"):
            auto_label(config, dataset.directory)
        assert obsolete.is_file()
        if failure_stage == "after_commit":
            committed = json.loads(report.read_text())
            assert committed["frames"] == record_count - 2
            assert all(Path(path).is_file() for path in committed["contact_sheets"])
        else:
            assert report.read_bytes() == old_report
        assert all(path.read_bytes() == content for path, content in original_sheets.items())
        assert legacy.exists()
        # A later successful rebuild retires both the old set and any complete
        # generation left unreferenced by an interrupted report publication.
        monkeypatch.setattr("dmotion.autolabel._contact_sheets", render)
        monkeypatch.setattr("dmotion.autolabel._write_json", _write_json)
        state["rebuilding"] = False
        recovered = json.loads(auto_label(config, dataset.directory).read_text())
        assert set((output / "contact-sheets").glob("*/contact-sheet-*.jpg")) == {
            Path(path) for path in recovered["contact_sheets"]
        }
        assert not legacy.exists()
    else:
        summary = json.loads(auto_label(config, dataset.directory).read_text())
        assert summary["frames"] == summary["preserved_frames"] == record_count - 2
        assert len(summary["contact_sheets"]) == (record_count - 2 + 11) // 12
        current = Path(summary["contact_sheets"][0])
        assert current.name == "contact-sheet-01.jpg"
        assert current.parent.parent == output / "contact-sheets"
        assert not obsolete.exists()
        assert not legacy.exists()
        sheet = "\n".join(Path(path).read_text() for path in summary["contact_sheets"])
        assert all(record["id"] not in sheet for record in reviewed)
    assert notes.read_bytes() == b"user notes"
    assert generation_notes.read_bytes() == b"generation notes"
    assert dataset.manifest_path.read_bytes() == manifest
