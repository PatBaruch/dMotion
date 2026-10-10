import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dmotion import evaluation
from dmotion.config import AppConfig
from dmotion.dataset import Dataset
from dmotion.detector import Detection
from dmotion.evaluation import DEFAULT_GATE, calibrate, evaluate_checkpoints, score_predictions


def truth(id, status="positive"):
    return {
        "id": id,
        "status": status,
        "width": 100,
        "height": 100,
        "boxes": [[10, 10, 30, 30]] if status == "positive" else [],
    }


def prediction(confidence=0.9, box=None):
    return {"box": box or [10, 10, 30, 30], "confidence": confidence}


def test_matching_counts_duplicates_oversized_boxes_and_negative_frames():
    records = [truth("cash"), truth("empty", "negative")]
    predictions = {
        "cash": [prediction(), prediction(0.8), prediction(0.7, [0, 0, 100, 100])],
        "empty": [prediction(), prediction(0.6)],
    }
    score = score_predictions(records, predictions, 0.5)
    assert (score["true_positive_boxes"], score["false_positive_boxes"], score["missed_boxes"]) == (
        1,
        4,
        0,
    )
    assert score["cash_frames_found"] == 1
    assert score["false_alarm_frames"] == 1
    assert score["false_alarm_rate"] == 1
    assert score["precision"] == 0.2
    assert score["recall"] == 1


def test_matching_uses_confidence_order_and_never_reuses_a_truth_box():
    row = truth("cash")
    row["boxes"] = [[0, 0, 20, 20], [10, 0, 30, 20]]
    score = score_predictions(
        [row], {"cash": [prediction(0.8, [0, 0, 20, 20]), prediction(0.9, [0, 0, 30, 20])]}, 0.5
    )
    assert score["true_positive_boxes"] == 1
    assert score["missed_boxes"] == 1
    assert score["false_positive_boxes"] == 1


def test_missing_negatives_cannot_pass_gate_and_empty_predictions_are_misses():
    score = score_predictions([truth("cash")], {"cash": []}, 0.5)
    assert score["missed_boxes"] == 1
    assert score["false_alarm_rate"] is None
    assert "No negative frames" in evaluation.gate_failures(score, DEFAULT_GATE)


def test_calibration_prefers_passing_threshold_then_f1_recall_and_higher_threshold():
    records = [truth("cash"), truth("empty", "negative")]
    predictions = {"cash": [prediction(0.7)], "empty": [prediction(0.2)]}
    result = calibrate(records, predictions, [0.1, 0.5, 0.6, 0.8], DEFAULT_GATE, 0.5)
    assert result["gate_passed"]
    assert result["selected"]["threshold"] == 0.6
    assert result["failures"] == []


@pytest.mark.parametrize(
    "confidence,box",
    [
        (float("nan"), None),
        (True, None),
        (1.1, None),
        (0.9, [0, 0, 0, 1]),
        (0.9, [0, 0, 101, 1]),
        (0.9, [0, 0, float("inf"), 1]),
    ],
)
def test_invalid_predictions_fail_instead_of_reporting_plausible_metrics(confidence, box):
    with pytest.raises(ValueError):
        score_predictions([truth("cash")], {"cash": [prediction(confidence, box)]}, 0.5)


@pytest.fixture
def setup(tmp_path):
    dataset = Dataset(tmp_path / "source")
    for group in ("training", "validation", "testing"):
        for status in ("positive", "negative"):
            image = tmp_path / f"{group}-{status}.jpg"
            image.write_bytes(f"{group} {status}".encode())
            row = dataset.add_image(image, group=group, width=100, height=100)
            dataset.review(row["id"], status=status, boxes=truth("unused", status)["boxes"])
    splits = {"training": "train", "validation": "val", "testing": "test"}
    split_file = tmp_path / "splits.json"
    split_file.write_text(json.dumps(splits))
    baseline, candidate = tmp_path / "baseline.pt", tmp_path / "candidate.pt"
    baseline.write_bytes(b"baseline")
    candidate.write_bytes(b"candidate")
    return SimpleNamespace(
        config=AppConfig(root=tmp_path),
        dataset=dataset,
        splits=splits,
        split_file=split_file,
        baseline=baseline,
        candidate=candidate,
        output=tmp_path / "evaluation",
    )


def run(setup, **kwargs):
    return evaluate_checkpoints(
        setup.config,
        setup.dataset.directory,
        setup.split_file,
        setup.candidate,
        baseline=setup.baseline,
        output=setup.output,
        thresholds=[0.1, 0.5, 0.8],
        **kwargs,
    )


def fake_inference(monkeypatch, setup, *, bad_validation=False, bad_test=False, on_predict=None):
    calls = []

    class Detector:
        def __init__(self, config):
            self.kind = Path(config.detector.model).read_bytes().decode()
            self.model_sha256 = hashlib.sha256(self.kind.encode()).hexdigest()
            self.device = config.detector.device

    def predict(detector, dataset, records, *, warmup=False):
        split = setup.splits[records[0]["group"]]
        calls.append((detector.kind, split))
        if split == "test":
            receipt = json.loads((setup.output / "calibration.json").read_text())
            assert receipt["models"]["candidate"]["validation"]["selected"]["threshold"] == 0.5
            assert receipt["models"]["baseline"]["validation"]["selected"]["threshold"] == 0.8
            assert receipt["models"]["candidate"]["test"] is None
        if on_predict:
            on_predict(detector, dataset, records)
        result = {}
        for row in records:
            bad = detector.kind == "candidate" and (bad_validation if split == "val" else bad_test)
            result[row["id"]] = (
                []
                if row["status"] == "negative" or bad
                else [prediction(0.7 if detector.kind == "candidate" else 0.95)]
            )
        return result, {
            "device": detector.device,
            "mean_ms": 10,
            "median_ms": 9,
            "p95_ms": 12,
            "warmup_ms": 25 if warmup else None,
        }

    monkeypatch.setattr(evaluation, "MoneyDetector", Detector)
    monkeypatch.setattr(evaluation, "_predict_records", predict)
    return calls


def test_complete_comparison_freezes_thresholds_and_preserves_inputs(setup, monkeypatch):
    original = setup.dataset.manifest_path.read_bytes()
    calls = fake_inference(monkeypatch, setup)
    report = json.loads(run(setup).read_text())
    assert calls == [
        ("baseline", "val"),
        ("candidate", "val"),
        ("baseline", "test"),
        ("candidate", "test"),
    ]
    assert report["candidate_gate_passed"]
    assert report["verdict"] == "passed_frame_gates"
    assert report["active_model_replaced"] is False
    assert report["test_context_declared"] == "previously-inspected"
    assert (
        report["models"]["candidate"]["model_revision"]
        == "sha256:" + hashlib.sha256(b"candidate").hexdigest()
    )
    assert setup.dataset.manifest_path.read_bytes() == original
    assert setup.candidate.read_bytes() == b"candidate"
    assert (setup.output / "source-manifest.json").read_bytes() == original
    assert len(Dataset(setup.output / "dataset").records()) == 4
    assert not list((setup.output / ".checkpoints").glob("checkpoint-*"))
    assert (setup.output / "report.md").is_file()
    assert json.loads((setup.output / "run.json").read_text())["stage"] == "complete"


def test_validation_failure_does_not_score_test_or_report_promotion(setup, monkeypatch):
    calls = fake_inference(monkeypatch, setup, bad_validation=True)
    report = json.loads(run(setup).read_text())
    assert calls == [("baseline", "val"), ("candidate", "val")]
    assert not report["candidate_gate_passed"]
    assert report["verdict"] == "failed_validation_test_not_run"
    assert report["models"]["candidate"]["test"] is None
    assert "test" not in json.loads((setup.output / "predictions.json").read_text())["candidate"]
    assert "Box recall is below the minimum" in (setup.output / "report.md").read_text()


def test_test_failure_does_not_recalibrate_on_test(setup, monkeypatch):
    fake_inference(monkeypatch, setup, bad_test=True)
    report = json.loads(run(setup).read_text())
    assert report["verdict"] == "failed_test"
    assert report["models"]["candidate"]["test"]["metrics"]["threshold"] == 0.5
    assert report["models"]["candidate"]["test"]["metrics"]["recall"] == 0


def test_atomic_source_changes_do_not_change_frozen_labels_or_model_identity(setup, monkeypatch):
    def change_source(detector, dataset, records):
        assert all(
            dataset.image_path(r).read_bytes() == setup.dataset.image_path(r).read_bytes()
            for r in records
        )
        setup.candidate.write_bytes(b"replacement model")
        row = setup.dataset.records()[0]
        setup.dataset.review(row["id"], status="excluded", boxes=[])

    fake_inference(monkeypatch, setup, on_predict=change_source)
    report = json.loads(run(setup).read_text())
    assert report["models"]["candidate"]["model_sha256"] == hashlib.sha256(b"candidate").hexdigest()
    assert all(r["status"] != "excluded" for r in Dataset(setup.output / "dataset").records())


@pytest.mark.parametrize(
    "option,value",
    [
        ("thresholds", []),
        ("thresholds", [float("nan")]),
        ("thresholds", [True]),
        ("min_precision", -0.1),
        ("min_recall", float("inf")),
        ("max_false_alarm_rate", 1.1),
        ("iou_threshold", 0),
        ("test_context", "unverified"),
    ],
)
def test_invalid_settings_fail_before_model_load_or_output(setup, monkeypatch, option, value):
    monkeypatch.setattr(evaluation, "MoneyDetector", lambda _: pytest.fail("Model loaded"))
    kwargs = {"thresholds": [0.5], option: value}
    with pytest.raises(ValueError):
        evaluate_checkpoints(
            setup.config,
            setup.dataset.directory,
            setup.split_file,
            setup.candidate,
            baseline=setup.baseline,
            output=setup.output,
            **kwargs,
        )
    assert not setup.output.exists()


@pytest.mark.parametrize(
    "problem", ["unreviewed", "missing-group", "bad-split", "no-negative", "changed-image"]
)
def test_dataset_integrity_blocks_model_loading(setup, monkeypatch, problem):
    if problem == "unreviewed":
        row = setup.dataset.records()[0]
        setup.dataset.review(row["id"], status="unreviewed", boxes=[])
    elif problem == "missing-group":
        setup.splits.pop("testing")
    elif problem == "bad-split":
        setup.splits["testing"] = "validation"
    elif problem == "no-negative":
        row = next(
            r
            for r in setup.dataset.records()
            if r["group"] == "testing" and r["status"] == "negative"
        )
        setup.dataset.review(row["id"], status="excluded", boxes=[])
    else:
        row = next(r for r in setup.dataset.records() if r["group"] == "validation")
        setup.dataset.image_path(row).write_bytes(b"changed")
    setup.split_file.write_text(json.dumps(setup.splits))
    monkeypatch.setattr(evaluation, "MoneyDetector", lambda _: pytest.fail("Model loaded"))
    with pytest.raises(ValueError):
        run(setup)
    assert not (setup.output / "report.json").exists()
    assert json.loads((setup.output / "run.json").read_text())["stage"] == "failed"


def test_export_plan_checks_record_identity_labels_and_session_boundaries(setup):
    records = setup.dataset.records()
    rows = [{**r, "split": setup.splits[r["group"]]} for r in records]
    assert evaluation._split_plan({"records": rows}, records) == setup.splits
    rows[0]["boxes"] = [[0, 0, 10, 10]]
    with pytest.raises(ValueError, match="provenance"):
        evaluation._split_plan({"records": rows}, records)
    rows = [{**r, "split": setup.splits[r["group"]]} for r in records]
    rows[0]["split"] = "val"
    with pytest.raises(ValueError, match="more than one"):
        evaluation._split_plan({"records": rows}, records)
    with pytest.raises(ValueError, match="match"):
        evaluation._split_plan({"records": rows[:-1]}, records)


def test_existing_outputs_and_source_overlap_are_never_overwritten(setup):
    setup.output.mkdir()
    keep = setup.output / "important.txt"
    keep.write_text("keep")
    with pytest.raises(FileExistsError):
        run(setup)
    assert keep.read_text() == "keep"
    with pytest.raises(ValueError, match="separate"):
        evaluate_checkpoints(
            setup.config,
            setup.dataset.directory,
            setup.split_file,
            setup.candidate,
            baseline=setup.baseline,
            output=setup.dataset.directory / "report",
        )


def test_missing_dataset_does_not_create_a_manifest(setup):
    directory = setup.config.root / "typo"
    with pytest.raises(ValueError, match="existing reviewed"):
        evaluate_checkpoints(setup.config, directory, setup.split_file, setup.candidate)
    assert not directory.exists()


def test_loaded_checkpoint_hash_mismatch_fails_without_report(setup, monkeypatch):
    monkeypatch.setattr(
        evaluation, "MoneyDetector", lambda _: SimpleNamespace(model_sha256="wrong")
    )
    with pytest.raises(ValueError, match="provenance"):
        run(setup)
    assert not (setup.output / "report.json").exists()


def test_inference_decodes_verified_bytes_and_separates_warmup(setup, monkeypatch):
    rows = [r for r in setup.dataset.records() if r["group"] == "validation"]
    seen = []
    monkeypatch.setitem(
        sys.modules, "numpy", SimpleNamespace(uint8="uint8", frombuffer=lambda b, **k: b)
    )
    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(
            IMREAD_COLOR=1, imdecode=lambda b, _: SimpleNamespace(shape=(100, 100, 3), data=b)
        ),
    )

    def predict(frame):
        seen.append(frame.data)
        return [Detection((10, 10, 30, 30), 0.9, "cash")]

    detector = SimpleNamespace(device="cpu", predict=predict)
    predictions, timing = evaluation._predict_records(detector, setup.dataset, rows, warmup=True)
    assert len(seen) == 3
    assert seen[0] == seen[1] == setup.dataset.image_path(rows[0]).read_bytes()
    assert len(predictions) == 2
    assert timing["warmup_ms"] is not None
    assert timing["images"] == 2
    setup.dataset.image_path(rows[0]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        evaluation._predict_records(detector, setup.dataset, rows)


def test_inference_rejects_decode_failure_and_unrelated_classes(setup, monkeypatch):
    rows = [r for r in setup.dataset.records() if r["group"] == "validation"]
    monkeypatch.setitem(
        sys.modules, "numpy", SimpleNamespace(uint8="uint8", frombuffer=lambda b, **k: b)
    )
    cv2 = SimpleNamespace(IMREAD_COLOR=1, imdecode=lambda b, _: None)
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    detector = SimpleNamespace(
        device="cpu", predict=lambda _: [Detection((10, 10, 30, 30), 0.9, "person")]
    )
    with pytest.raises(ValueError, match="decode"):
        evaluation._predict_records(detector, setup.dataset, rows)
    cv2.imdecode = lambda b, _: SimpleNamespace(shape=(100, 100, 3))
    with pytest.raises(ValueError, match="single-class"):
        evaluation._predict_records(detector, setup.dataset, rows)


@pytest.mark.parametrize("failed_file", ["run.json", "report.json"])
def test_late_publication_error_never_leaves_a_completed_report(setup, monkeypatch, failed_file):
    fake_inference(monkeypatch, setup)
    write = evaluation._write_json

    def fail_once(path, value):
        if path.name == failed_file and value.get("stage") == "complete":
            raise OSError("Simulated publication failure")
        write(path, value)

    monkeypatch.setattr(evaluation, "_write_json", fail_once)
    with pytest.raises(OSError, match="publication"):
        run(setup)
    assert not (setup.output / "report.json").exists()
    assert json.loads((setup.output / "run.json").read_text())["stage"] == "failed"
