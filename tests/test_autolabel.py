import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from filelock import FileLock, Timeout

from dmotion.autolabel import _contact_sheets, _draft_boxes, auto_label
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
        model_sha256 = hashlib.sha256(b"test detector weights").hexdigest()
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
    assert summary["model_sha256"] == Detector.model_sha256
    assert summary["model_revision"] == "sha256:" + Detector.model_sha256
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
    assert cash["suggestion"]["model_sha256"] == Detector.model_sha256
    assert empty["suggestion"]["model_sha256"] == Detector.model_sha256


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
        lambda config: SimpleNamespace(
            device="cpu", model=SimpleNamespace(), model_sha256="a" * 64
        ),
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


@pytest.mark.parametrize("fail_report", [False, True])
def test_final_human_review_publishes_empty_audit_and_preserves_truth(
    tmp_path, monkeypatch, fail_report
):
    dataset = Dataset(tmp_path / "dataset")
    for index in range(3):
        image = tmp_path / f"frame-{index}.jpg"
        image.write_bytes(f"image {index}".encode())
        record = dataset.add_image(image, group="session", width=100, height=80)
        dataset.suggest(
            record["id"],
            {
                "boxes": [],
                "scores": [],
                "labels": [],
                "model": "cash-checkpoint",
                "created_at": "2026-10-07T00:00:00+00:00",
                "prompts": ["cash"],
            },
        )

    def render(dataset, records, output):
        path = output / "contact-sheet-01.jpg"
        path.write_text("\n".join(record["id"] for record in records))
        return [str(path)]

    monkeypatch.setattr("dmotion.autolabel._contact_sheets", render)
    config = AppConfig(root=tmp_path)
    report = auto_label(config, dataset.directory)
    previous = report.read_bytes()
    sheet = Path(json.loads(previous)["contact_sheets"][0])
    sheet_bytes = sheet.read_bytes()
    notes = sheet.parent / "review-notes.txt"
    notes.write_bytes(b"human notes")
    for record, status in zip(dataset.records(), ["positive", "negative", "excluded"], strict=True):
        dataset.review(
            record["id"], status=status, boxes=[[2, 3, 12, 13]] if status == "positive" else []
        )
    manifest = dataset.manifest_path.read_bytes()
    images = {path: path.read_bytes() for path in (dataset.directory / "images").iterdir()}
    monkeypatch.setattr("dmotion.autolabel._contact_sheets", _contact_sheets)
    for module in ("PIL", "cv2", "ultralytics"):
        monkeypatch.setitem(sys.modules, module, None)
    if fail_report:

        def write(path, value):
            if path.name == "report.json":
                raise RuntimeError("report interrupted")
            _write_json(path, value)

        monkeypatch.setattr("dmotion.autolabel._write_json", write)
        with pytest.raises(RuntimeError, match="report interrupted"):
            auto_label(config, dataset.directory)
        assert report.read_bytes() == previous
        assert sheet.read_bytes() == sheet_bytes
        monkeypatch.setattr("dmotion.autolabel._write_json", _write_json)
    summary = json.loads(auto_label(config, dataset.directory).read_text())
    for field in ("frames", "new_frames", "preserved_frames", "frames_with_boxes", "boxes"):
        assert summary[field] == 0
    assert summary["contact_sheets"] == summary["models"] == []
    assert json.loads(report.with_name("predictions.json").read_text())["records"] == []
    assert not sheet.exists()
    assert not list(report.parent.glob("contact-sheets/*/contact-sheet-*.jpg"))
    assert notes.read_bytes() == b"human notes"
    assert dataset.manifest_path.read_bytes() == manifest
    assert all(path.read_bytes() == contents for path, contents in images.items())


def test_empty_dataset_publishes_empty_audit_without_optional_vision(tmp_path, monkeypatch):
    dataset = Dataset(tmp_path / "dataset")
    for module in ("PIL", "cv2", "ultralytics"):
        monkeypatch.setitem(sys.modules, module, None)
    summary = json.loads(auto_label(AppConfig(root=tmp_path), dataset.directory).read_text())
    assert summary["frames"] == 0
    assert summary["contact_sheets"] == []


@pytest.mark.parametrize("review_all", [False, True])
def test_publication_reconciles_late_reviews_and_locks_the_manifest(
    tmp_path, monkeypatch, review_all
):
    dataset = Dataset(tmp_path / "dataset")
    for index in range(2):
        image = tmp_path / f"frame-{index}.jpg"
        image.write_bytes(f"image {index}".encode())
        record = dataset.add_image(image, group="session", width=100, height=80)
        dataset.suggest(
            record["id"],
            {
                "boxes": [],
                "scores": [],
                "labels": [],
                "model": f"saved-model-{index}",
                "created_at": "2026-10-07T00:00:00+00:00",
                "prompts": ["cash"],
            },
        )
    original = dataset.records()
    other_lock = FileLock(dataset._manifest_lock.lock_file)
    reviewed = original if review_all else original[:1]
    state = {"reviewed": False}

    def write(path, value):
        _write_json(path, value)
        if path.name == "predictions.json" and not state["reviewed"]:
            state["reviewed"] = True
            for record in reviewed:
                dataset.review(record["id"], status="positive", boxes=[[2, 3, 12, 13]])
            state["manifest"] = dataset.manifest_path.read_bytes()

    def render(dataset, records, output):
        state["rendered"] = [record["id"] for record in records]
        # A competing labeler/importer cannot mutate the manifest between this
        # final snapshot and report publication. No threads or timing sleeps needed.
        with pytest.raises(Timeout), other_lock.acquire(timeout=0):
            pass
        return []

    monkeypatch.setitem(sys.modules, "cv2", None)
    monkeypatch.setattr("dmotion.autolabel._write_json", write)
    monkeypatch.setattr("dmotion.autolabel._contact_sheets", render)
    report = auto_label(AppConfig(root=tmp_path), dataset.directory)
    summary = json.loads(report.read_text())
    expected = [] if review_all else [original[1]["id"]]
    assert state["rendered"] == expected
    predictions = json.loads(report.with_name("predictions.json").read_text())["records"]
    assert [record["id"] for record in predictions] == expected
    assert summary["frames"] == summary["preserved_frames"] == len(expected)
    assert summary["new_frames"] == 0
    assert [model["model"] for model in summary["models"]] == (
        [] if review_all else ["saved-model-1"]
    )
    assert dataset.manifest_path.read_bytes() == state["manifest"]
    with other_lock.acquire(timeout=0):
        pass  # Publication released the manifest lock.


@pytest.mark.parametrize("decision", ["positive", "negative", "excluded", "all", "other-draft"])
def test_inference_preserves_a_concurrent_review_or_existing_draft(tmp_path, monkeypatch, decision):
    dataset = Dataset(tmp_path / "dataset")
    records = []
    for index in range(2):
        image = tmp_path / f"frame-{index}.jpg"
        image.write_bytes(f"image {index}".encode())
        records.append(dataset.add_image(image, group="session", width=100, height=80))
    checkpoint = tmp_path / "models/cash-yolo26m.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    state = {"calls": 0}

    class Detector:
        model_sha256 = hashlib.sha256(b"test detector weights").hexdigest()
        device = "cpu"

        def __init__(self, config):
            pass

        def predict(self, frame):
            state["calls"] += 1
            if state["calls"] == 1:
                changed = records if decision == "all" else records[:1]
                for record in changed:
                    if decision == "other-draft":
                        dataset.suggest(
                            record["id"],
                            {
                                "boxes": [],
                                "scores": [],
                                "labels": [],
                                "model": "other-checkpoint",
                                "created_at": "2026-10-07T00:00:00+00:00",
                                "prompts": ["cash"],
                            },
                        )
                    else:
                        dataset.review(
                            record["id"],
                            status="negative" if decision == "all" else decision,
                            boxes=[[2, 3, 12, 13]] if decision == "positive" else [],
                        )
                state["preserved"] = [dataset.get_record(record["id"]) for record in changed]
            return [Detection((10, 10, 90, 70), 0.8, "cash")]

    monkeypatch.setitem(
        sys.modules, "cv2", SimpleNamespace(imread=lambda _: SimpleNamespace(shape=(80, 100, 3)))
    )
    monkeypatch.setattr("dmotion.autolabel.MoneyDetector", Detector)
    monkeypatch.setattr("dmotion.autolabel._contact_sheets", lambda *args: [])
    report = auto_label(AppConfig(root=tmp_path), dataset.directory)
    summary = json.loads(report.read_text())
    expected = [] if decision == "all" else records if decision == "other-draft" else records[1:]
    published = json.loads(report.with_name("predictions.json").read_text())["records"]
    assert [record["id"] for record in published] == [record["id"] for record in expected]
    assert all(record["status"] == "unreviewed" for record in published)
    assert summary["frames"] == len(expected)
    assert summary["new_frames"] == (0 if decision == "all" else 1)
    assert summary["preserved_frames"] == (1 if decision == "other-draft" else 0)
    for record in state["preserved"]:
        assert dataset.get_record(record["id"]) == record
    if decision == "other-draft":
        assert published[0]["suggestion"]["model"] == "other-checkpoint"


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
        model_sha256 = hashlib.sha256(b"test detector weights").hexdigest()
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
