import json
from types import SimpleNamespace

import pytest
from test_training import fake_dependencies, small_dataset

from dmotion.config import AppConfig
from dmotion.dataset import Dataset, export_dataset
from dmotion.harness import calibrate, passes_gate, run_harness, score_predictions


def examples():
    return [
        {"id": "cash", "status": "positive", "boxes": [[10, 10, 30, 30]]},
        {"id": "shirt", "status": "negative", "boxes": []},
    ]


def test_matching_rejects_person_boxes_and_duplicate_cash_boxes():
    scores = score_predictions(
        examples(),
        {
            "cash": [
                {"box": [0, 0, 100, 100], "confidence": 0.9},
                {"box": [10, 10, 30, 30], "confidence": 0.8},
                {"box": [10, 10, 30, 30], "confidence": 0.7},
            ],
            "shirt": [{"box": [0, 0, 100, 100], "confidence": 0.8}],
        },
        0.25,
    )
    assert scores["true_positive_boxes"] == 1
    assert scores["false_positive_boxes"] == 3
    assert scores["false_alarm_frames"] == 1
    assert scores["cash_frames_found"] == 1


def test_calibration_prefers_threshold_that_keeps_cash_and_rejects_false_alarms():
    selection = calibrate(
        examples(),
        {
            "cash": [{"box": [10, 10, 30, 30], "confidence": 0.7}],
            "shirt": [{"box": [0, 0, 100, 100], "confidence": 0.4}],
        },
        thresholds=(0.25, 0.5, 0.8),
    )
    assert selection["gate_passed"]
    assert selection["selected"]["threshold"] == 0.5
    assert selection["selected"]["false_alarm_rate"] == 0
    assert selection["selected"]["recall"] == 1


def test_gate_does_not_pass_without_measured_negatives_or_when_everything_is_missed():
    metrics = score_predictions(examples()[:1], {"cash": []}, 0.5)
    assert not passes_gate(metrics, max_false_alarm_rate=0.05, min_precision=0.8, min_recall=0.5)
    result = calibrate(examples(), {"cash": [], "shirt": []})
    assert not result["gate_passed"]


@pytest.mark.parametrize("thresholds", [(), (float("nan"),), (0,), (1.1,)])
def test_invalid_calibration_thresholds(thresholds):
    with pytest.raises(ValueError, match="thresholds"):
        calibrate(examples(), {}, thresholds=thresholds)


def reviewed_dataset(tmp_path):
    dataset = small_dataset(tmp_path)
    for i in range(3):
        path = tmp_path / f"negative-{i}.jpg"
        path.write_bytes(f"negative {i}".encode())
        record = dataset.add_image(path, group=f"shoot-{i}", width=100, height=100)
        dataset.review(record["id"], status="negative", boxes=[], reviewer="test-reviewer")
    return dataset, {"shoot-0": "train", "shoot-1": "val", "shoot-2": "test"}


def test_explicit_export_preserves_all_records_of_each_session(tmp_path):
    dataset, splits = reviewed_dataset(tmp_path)
    path = export_dataset(dataset, tmp_path / "export", group_splits=splits)
    report = json.loads(path.with_name("export-report.json").read_text())
    for record in report["records"]:
        assert record["split"] == splits[record["group"]]
    assert all(split["negative"] == 1 for split in report["splits"].values())


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"shoot-0": "train"},
        {"shoot-0": "train", "shoot-1": "train", "shoot-2": "train"},
        {"shoot-0": "train", "shoot-1": "val", "shoot-2": "test", "extra": "train"},
    ],
)
def test_bad_split_plans_fail_before_export(tmp_path, bad):
    dataset, _ = reviewed_dataset(tmp_path)
    with pytest.raises(ValueError, match="Split"):
        export_dataset(dataset, tmp_path / "export", group_splits=bad)
    assert not (tmp_path / "export").exists()


def test_candidate_training_defers_test_and_preserves_active_weights(tmp_path, monkeypatch):
    dataset, splits = reviewed_dataset(tmp_path)
    calls = fake_dependencies(monkeypatch)
    active = tmp_path / "models/money-spread.pt"
    active.parent.mkdir()
    active.write_bytes(b"active")
    from dmotion.training import train_model

    path = train_model(
        AppConfig(root=tmp_path),
        dataset.directory,
        epochs=1,
        candidate_directory=tmp_path / "candidate",
        group_splits=splits,
        evaluate_test=False,
    )
    assert path.name == "candidate.pt" and active.read_bytes() == b"active"
    assert not any(event == "val" for event, _ in calls)
    assert json.loads(path.with_suffix(".json").read_text())["candidate_only"]


@pytest.mark.parametrize("continue_checkpoint", [False, True])
def test_harness_freezes_dataset_and_calibrates_before_using_test(
    tmp_path, monkeypatch, continue_checkpoint
):
    dataset, splits = reviewed_dataset(tmp_path)
    split_file = tmp_path / "splits.json"
    split_file.write_text(json.dumps(splits))
    baseline = tmp_path / "active.pt"
    baseline.write_bytes(b"original")
    events = []

    def train(config, directory, **kwargs):
        assert not kwargs["evaluate_test"]
        if continue_checkpoint:
            assert kwargs["initial_model"] == tmp_path / "run/initial.pt"
            assert kwargs["initial_model"].read_bytes() == b"original"
        assert kwargs["group_splits"] == splits
        candidate = kwargs["candidate_directory"] / "candidate.pt"
        candidate.parent.mkdir()
        candidate.write_bytes(b"candidate")
        return candidate

    class Detector:
        device = "cpu"

        def __init__(self, config):
            events.append(config.detector.model)

    def predict(detector, frozen, records):
        events.append(records[0]["group"])
        assert frozen.directory != dataset.directory
        return (
            {
                r["id"]: ([{"box": r["boxes"][0], "confidence": 0.6}] if r["boxes"] else [])
                for r in records
            },
            {"device": "cpu"},
        )

    monkeypatch.setattr("dmotion.harness.train_model", train)
    monkeypatch.setattr("dmotion.harness.MoneyDetector", Detector)
    monkeypatch.setattr("dmotion.harness._predict_records", predict)
    path = run_harness(
        AppConfig(root=tmp_path),
        dataset.directory,
        split_file,
        baseline=baseline,
        output=tmp_path / "run",
        initial_model=baseline if continue_checkpoint else None,
    )
    report = json.loads(path.read_text())
    assert report["quality_gate_passed"] and not report["active_model_replaced"]
    assert report["promotion_eligible"] is not continue_checkpoint
    assert report["initial_model_data_overlap_unknown"] is continue_checkpoint
    assert baseline.read_bytes() == b"original"
    assert events[1:3] == ["shoot-1", "shoot-2"]
    assert events[4:6] == ["shoot-1", "shoot-2"]
    assert report["models"]["candidate"]["test"]["recall"] == 1
    first = dataset.records()[0]
    dataset.review(first["id"], status="excluded", boxes=[])
    assert Dataset(path.parent / "dataset").get_record(first["id"])["status"] == "positive"


def test_unreviewed_frames_are_never_converted_to_negatives(tmp_path):
    dataset, splits = reviewed_dataset(tmp_path)
    source = tmp_path / "pending.jpg"
    source.write_bytes(b"pending")
    dataset.add_image(source, group="new", width=100, height=100)
    split_file = tmp_path / "splits.json"
    split_file.write_text(json.dumps(splits))
    with pytest.raises(ValueError, match="unreviewed"):
        run_harness(AppConfig(root=tmp_path), dataset.directory, split_file)
    assert dataset.summary()["unreviewed"] == 1
    assert not (tmp_path / "outputs").exists()


def test_comparison_uses_calibrated_threshold_and_original_validation_only(tmp_path, monkeypatch):
    from dmotion.harness import compare_detector

    dataset, splits = reviewed_dataset(tmp_path)
    model = tmp_path / "model.pt"
    model.write_bytes(b"weights")
    calls = []
    monkeypatch.setattr("dmotion.harness.MoneyDetector", lambda c: SimpleNamespace(device="cpu"))

    def predict(detector, data, records):
        group = records[0]["group"]
        calls.append(group)
        if group == "shoot-2":
            assert (tmp_path / "report/calibration.json").is_file()
        return (
            {
                r["id"]: ([{"box": r["boxes"][0], "confidence": 0.4}] if r["boxes"] else [])
                for r in records
            },
            {},
        )

    monkeypatch.setattr("dmotion.harness._predict_records", predict)
    result = compare_detector(AppConfig(root=tmp_path), dataset, splits, model, tmp_path / "report")
    assert calls == ["shoot-1", "shoot-2"]
    assert result["test"]["threshold"] == result["validation"]["selected"]["threshold"]


def test_labeler_comparison_requires_reviewed_matching_frame_sets(tmp_path):
    from dmotion.harness import compare_label_drafts

    dataset, _ = reviewed_dataset(tmp_path)
    records = dataset.records()
    files = []
    for i in range(2):
        rows = []
        for record in records:
            suggestion = {
                "boxes": record["boxes"],
                "scores": [0.8] * len(record["boxes"]),
                "labels": ["cash"] * len(record["boxes"]),
                "model": f"teacher-{i}",
                "created_at": "2026-10-04",
                "prompts": ["banknotes"],
            }
            rows.append({**record, "suggestion": suggestion})
        file = tmp_path / f"teacher-{i}.json"
        file.write_text(json.dumps({"records": rows}))
        files.append(file)
    output = tmp_path / "comparison.json"
    compare_label_drafts(dataset.directory, files, output)
    result = json.loads(output.read_text())
    assert all(t["metrics"]["precision"] == 1 for t in result["teachers"])
    assert len(result["frame_ids"]) == 6
    rows.pop()
    files[1].write_text(json.dumps({"records": rows}))
    with pytest.raises(ValueError, match="same frame IDs"):
        compare_label_drafts(dataset.directory, files, output)


def test_no_labeler_comparison_from_pending_drafts(tmp_path):
    from dmotion.harness import compare_label_drafts

    dataset, _ = reviewed_dataset(tmp_path)
    record = dataset.records()[0]
    dataset.review(record["id"], status="unreviewed", boxes=[])
    file = tmp_path / "predictions.json"
    file.write_text(json.dumps({"records": [record]}))
    with pytest.raises(ValueError, match="Review or exclude"):
        compare_label_drafts(dataset.directory, [file, file], tmp_path / "report.json")


def test_initial_checkpoint_is_used_and_hashed_without_replacing_it(tmp_path, monkeypatch):
    from dmotion.training import train_model

    dataset, splits = reviewed_dataset(tmp_path)
    calls = fake_dependencies(monkeypatch)
    initial = tmp_path / "cash.pt"
    initial.write_bytes(b"existing learned cash weights")
    candidate = train_model(
        AppConfig(root=tmp_path),
        dataset.directory,
        epochs=1,
        candidate_directory=tmp_path / "candidate",
        group_splits=splits,
        evaluate_test=False,
        initial_model=initial,
    )
    info = json.loads(candidate.with_suffix(".json").read_text())
    assert calls[0] == ("load", str(initial))
    assert initial.read_bytes() == b"existing learned cash weights"
    assert info["initial_model_override"] and info["pretrained_sha256"]


def test_missing_initial_checkpoint_fails_before_dataset_or_ml_imports(tmp_path):
    from dmotion.training import train_model

    with pytest.raises(ValueError, match="Initial model"):
        train_model(
            AppConfig(root=tmp_path), tmp_path / "dataset", initial_model=tmp_path / "missing.pt"
        )
    assert not (tmp_path / "dataset").exists()
