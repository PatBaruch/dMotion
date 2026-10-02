"""Keep AI draft boxes separate from the user's labels and training exports."""

import json
from copy import deepcopy

import pytest

from dmotion.dataset import Dataset, export_dataset


def add_sample(dataset, directory, name, *, group="session-one"):
    image = directory / f"{name}.jpg"
    image.write_bytes(f"distinct sample contents {name}".encode())
    return dataset.add_image(image, group=group, source=f"local:{name}", width=100, height=80)


def suggestion(boxes=None):
    boxes = [[10, 20, 50, 60]] if boxes is None else boxes
    return {
        "boxes": deepcopy(boxes),
        "scores": [0.8] * len(boxes),
        "labels": ["paper money"] * len(boxes),
        "model": "yolov8s-worldv2.pt",
        "prompts": ["paper money", "banknotes"],
        "created_at": "2026-10-02T09:00:00+00:00",
    }


def test_suggestions_stay_pending_and_do_not_enter_export_until_reviewed(tmp_path):
    dataset = Dataset(tmp_path / "dataset")
    reviewed_ids = set()
    for session in range(3):
        record = add_sample(dataset, tmp_path, f"reviewed-{session}", group=f"session-{session}")
        dataset.review(record["id"], status="positive", boxes=[[0, 0, 100, 80]])
        reviewed_ids.add(record["id"])

    cash = add_sample(dataset, tmp_path, "pending-cash", group="pending-cash")
    empty = add_sample(dataset, tmp_path, "pending-empty", group="pending-empty")
    proposed = dataset.suggest(cash["id"], suggestion())
    no_boxes = dataset.suggest(empty["id"], suggestion([]))
    for record in (proposed, no_boxes):
        assert record["status"] == "unreviewed"
        assert record["boxes"] == []
    assert dataset.summary()["ai_suggested"] == 2
    assert dataset.summary()["unreviewed"] == 2
    assert dataset.summary()["reviewed"] == 3

    yaml_path = export_dataset(dataset, tmp_path / "export")
    report = json.loads((yaml_path.parent / "export-report.json").read_text())
    assert {record["id"] for record in report["records"]} == reviewed_ids
    assert not list(yaml_path.parent.rglob(f"{cash['id']}*"))
    assert not list(yaml_path.parent.rglob(f"{empty['id']}*"))

    corrected_boxes = [[20, 10, 80, 70]]
    dataset.review(cash["id"], status="positive", boxes=corrected_boxes)
    dataset.review(empty["id"], status="negative", boxes=[])
    assert dataset.summary()["ai_suggested"] == 0
    assert dataset.summary()["unreviewed"] == 0
    assert dataset.summary()["reviewed"] == 5
    yaml_path = export_dataset(dataset, tmp_path / "export")
    report = json.loads((yaml_path.parent / "export-report.json").read_text())
    exported = {record["id"]: record for record in report["records"]}
    assert set(exported) == reviewed_ids | {cash["id"], empty["id"]}
    assert exported[cash["id"]]["boxes"] == corrected_boxes
    for record, expected in (
        (exported[cash["id"]], "0 0.50000000 0.50000000 0.60000000 0.75000000\n"),
        (exported[empty["id"]], ""),
    ):
        label = yaml_path.parent / "labels" / record["split"] / f"{record['id']}.txt"
        assert label.read_text() == expected


@pytest.mark.parametrize("status", ["positive", "negative", "excluded"])
def test_suggestions_cannot_replace_any_reviewed_record(tmp_path, status):
    dataset = Dataset(tmp_path / "dataset")
    record = add_sample(dataset, tmp_path, "sample")
    dataset.suggest(record["id"], suggestion())
    boxes = [[0, 0, 100, 80]] if status == "positive" else []
    reviewed = dataset.review(record["id"], status=status, boxes=boxes)
    before = dataset.manifest_path.read_bytes()
    with pytest.raises(ValueError, match="reviewed picture"):
        dataset.suggest(record["id"], suggestion([[20, 10, 80, 70]]))
    assert dataset.manifest_path.read_bytes() == before
    assert dataset.get_record(record["id"]) == reviewed
    assert dataset.summary()["ai_suggested"] == 0


def test_caller_mutation_does_not_change_persisted_suggestion(tmp_path):
    dataset = Dataset(tmp_path / "dataset")
    record = add_sample(dataset, tmp_path, "sample")
    draft = suggestion()
    expected = deepcopy(draft)
    saved = dataset.suggest(record["id"], draft)
    draft["boxes"][0][0] = 99
    draft["scores"][0] = 0.1
    draft["labels"][0] = "changed"
    draft["prompts"].clear()
    saved["suggestion"]["boxes"][0][0] = 98
    saved["suggestion"]["model"] = "another-model"
    saved["status"] = "positive"
    saved["boxes"] = [[0, 0, 100, 80]]
    persisted = Dataset(dataset.directory).get_record(record["id"])
    assert persisted["suggestion"] == expected
    assert persisted["status"] == "unreviewed"
    assert persisted["boxes"] == []


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("boxes", None),
        ("boxes", [[-1, 20, 50, 60]]),
        ("boxes", [[0, 0, 101, 80]]),
        ("boxes", [[10, 20, 10, 60]]),
        ("boxes", [[10.0, 20, 50, 60]]),
        ("boxes", [[True, 20, 50, 60]]),
        ("boxes", [[10, 20, 50]]),
        ("scores", []),
        ("scores", [0.5, 0.8]),
        ("scores", [True]),
        ("scores", ["0.8"]),
        ("scores", [-0.1]),
        ("scores", [1.1]),
        ("scores", [float("nan")]),
        ("scores", [float("inf")]),
        ("labels", []),
        ("labels", [""]),
        ("labels", [12]),
        ("model", " "),
        ("model", None),
        ("created_at", ""),
        ("created_at", 123),
        ("prompts", []),
        ("prompts", [" "]),
        ("prompts", [123]),
        ("prompts", "paper money"),
    ],
)
def test_invalid_suggestion_cannot_change_manifest(tmp_path, field, invalid):
    dataset = Dataset(tmp_path / "dataset")
    record = add_sample(dataset, tmp_path, "sample")
    dataset.suggest(record["id"], suggestion())
    before = dataset.manifest_path.read_bytes()
    draft = suggestion()
    draft[field] = invalid
    with pytest.raises(ValueError):
        dataset.suggest(record["id"], draft)
    assert dataset.manifest_path.read_bytes() == before


def test_empty_suggestion_remains_unreviewed(tmp_path):
    dataset = Dataset(tmp_path / "dataset")
    record = add_sample(dataset, tmp_path, "sample")
    saved = dataset.suggest(record["id"], suggestion([]))
    assert saved["status"] == "unreviewed"
    assert saved["boxes"] == []
    assert saved["suggestion"]["boxes"] == []
    assert saved["suggestion"]["scores"] == []
    assert saved["suggestion"]["labels"] == []
    summary = dataset.summary()
    assert summary["ai_suggested"] == summary["unreviewed"] == 1
    assert summary["reviewed"] == summary["negative"] == summary["positive"] == 0


def test_manifest_reader_validates_optional_suggestion(tmp_path):
    dataset = Dataset(tmp_path / "dataset")
    record = add_sample(dataset, tmp_path, "sample")
    assert "suggestion" not in dataset.get_record(record["id"])
    raw = json.loads(dataset.manifest_path.read_text())
    raw["records"][0]["suggestion"] = suggestion()
    raw["records"][0]["suggestion"]["scores"] = [1.1]
    dataset.manifest_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="scores"):
        Dataset(dataset.directory)
