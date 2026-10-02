import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dmotion.dataset import Dataset, export_dataset


def add_sample(dataset, directory, name, *, group="session-one"):
    image = directory / f"{name}.jpg"
    image.write_bytes(f"distinct sample contents {name}".encode())
    return dataset.add_image(image, group=group, source=f"local:{name}", width=100, height=80)


def reviewed_dataset(tmp_path):
    dataset = Dataset(tmp_path / "dataset")
    for session in range(3):
        group = f"recording-{session}"
        for frame in range(3):
            record = add_sample(dataset, tmp_path, f"positive-{session}-{frame}", group=group)
            dataset.review(record["id"], status="positive", boxes=[[10, 20, 50, 60]])
        record = add_sample(dataset, tmp_path, f"negative-{session}", group=group)
        dataset.review(record["id"], status="negative", boxes=[])
    return dataset


def test_import_deduplicates_bytes_without_changing_existing_review_or_group(tmp_path):
    dataset = Dataset(tmp_path / "dataset")
    record = add_sample(dataset, tmp_path, "original")
    reviewed = dataset.review(record["id"], status="positive", boxes=[[10, 20, 50, 60]])
    duplicate = tmp_path / "duplicate.png"
    duplicate.write_bytes((tmp_path / "original.jpg").read_bytes())
    assert dataset.add_image(duplicate, group="different-session", width=100, height=80) == reviewed
    assert dataset.summary()["total"] == 1
    assert dataset.image_path(reviewed).read_bytes() == duplicate.read_bytes()
    assert Dataset(dataset.directory).get_record(record["id"]) == reviewed


def test_import_normalizes_phone_orientation_before_recording_dimensions(tmp_path, monkeypatch):
    class PhoneImage:
        size = (100, 80)
        format = "JPEG"

        def __enter__(self):
            return self

        def __exit__(self, *arguments):
            pass

        def load(self):
            pass

        def getexif(self):
            return {274: 6}

    class NormalizedImage:
        size = (80, 100)

        def save(self, buffer, *, format):
            assert format == "JPEG"
            buffer.write(b"normalized pixels without orientation tag")

    monkeypatch.setitem(
        sys.modules,
        "PIL",
        SimpleNamespace(
            Image=SimpleNamespace(open=lambda path: PhoneImage()),
            ImageOps=SimpleNamespace(exif_transpose=lambda image: NormalizedImage()),
        ),
    )
    dataset = Dataset(tmp_path / "dataset")
    source = tmp_path / "phone.jpg"
    source.write_bytes(b"phone image with orientation tag")
    record = dataset.add_image(source, group="phone-session")
    assert (record["width"], record["height"]) == (80, 100)
    assert dataset.image_path(record).read_bytes() == b"normalized pixels without orientation tag"
    assert source.read_bytes() == b"phone image with orientation tag"
    assert dataset.add_image(source, group="another-group")["id"] == record["id"]


@pytest.mark.parametrize(
    ("status", "boxes"),
    [
        ("positive", []),
        ("positive", [[10, 10, 10, 20]]),
        ("positive", [[-1, 10, 20, 30]]),
        ("positive", [[0, 0, 101, 80]]),
        ("positive", [[0.5, 0, 20, 30]]),
        ("positive", [[True, 0, 20, 30]]),
        ("negative", [[0, 0, 20, 30]]),
        ("excluded", [[0, 0, 20, 30]]),
        ("unknown", []),
    ],
)
def test_invalid_review_does_not_change_manifest(tmp_path, status, boxes):
    dataset = Dataset(tmp_path / "dataset")
    record = add_sample(dataset, tmp_path, "sample")
    before = dataset.manifest_path.read_bytes()
    with pytest.raises(ValueError):
        dataset.review(record["id"], status=status, boxes=boxes)
    assert dataset.manifest_path.read_bytes() == before


def test_full_image_box_is_valid_and_caller_mutation_does_not_change_review(tmp_path):
    dataset = Dataset(tmp_path / "dataset")
    record = add_sample(dataset, tmp_path, "sample")
    boxes = [[0, 0, 100, 80]]
    reviewed = dataset.review(record["id"], status="positive", boxes=boxes)
    boxes[0][0] = 99
    reviewed["boxes"][0][0] = 98
    assert dataset.get_record(record["id"])["boxes"] == [[0, 0, 100, 80]]


def test_manifest_rejects_paths_outside_images(tmp_path):
    dataset = Dataset(tmp_path / "dataset")
    record = add_sample(dataset, tmp_path, "sample")
    raw = json.loads(dataset.manifest_path.read_text())
    raw["records"][0]["image"] = "../sample.jpg"
    dataset.manifest_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="inside"):
        dataset.get_record(record["id"])


def test_export_only_reviewed_images_and_split_complete_sessions(tmp_path):
    dataset = reviewed_dataset(tmp_path)
    pending = add_sample(dataset, tmp_path, "pending", group="web-source")
    excluded = add_sample(dataset, tmp_path, "excluded", group="web-source")
    dataset.review(excluded["id"], status="excluded", boxes=[])
    for session in range(3):
        record = add_sample(
            dataset, tmp_path, f"background-{session}", group=f"background-{session}"
        )
        dataset.review(record["id"], status="negative", boxes=[])
    yaml_path = export_dataset(dataset, tmp_path / "export")
    assert "money_spread" in yaml_path.read_text()
    assert str(yaml_path.parent) in yaml_path.read_text()
    report = json.loads((yaml_path.parent / "export-report.json").read_text())
    assert len(report["records"]) == 15
    assert pending["id"] not in {record["id"] for record in report["records"]}
    assert excluded["id"] not in {record["id"] for record in report["records"]}
    groups = {}
    for record in report["records"]:
        assert record["group"] not in groups or groups[record["group"]] == record["split"]
        groups[record["group"]] = record["split"]
        label = (yaml_path.parent / "labels" / record["split"] / f"{record['id']}.txt").read_text()
        if record["status"] == "positive":
            assert label == "0 0.30000000 0.50000000 0.40000000 0.50000000\n"
        else:
            assert label == ""
        assert (
            yaml_path.parent / "images" / record["split"] / Path(record["image"]).name
        ).is_file()
    assert all(
        count["positive"] == 3 and count["negative"] == 2 for count in report["splits"].values()
    )
    second = export_dataset(dataset, tmp_path / "export-two")
    report_two = json.loads((second.parent / "export-report.json").read_text())
    assert report["records"] == report_two["records"]


def test_export_refuses_frames_from_only_two_positive_sessions(tmp_path):
    dataset = Dataset(tmp_path / "dataset")
    for session in range(2):
        for frame in range(5):
            record = add_sample(dataset, tmp_path, f"{session}-{frame}", group=f"session-{session}")
            dataset.review(record["id"], status="positive", boxes=[[0, 0, 100, 80]])
    with pytest.raises(ValueError, match="at least 3 independent groups"):
        export_dataset(dataset, tmp_path / "export")


def test_export_replaces_old_generated_files_and_preserves_source_dataset(tmp_path):
    dataset = reviewed_dataset(tmp_path)
    output = tmp_path / "export"
    export_dataset(dataset, output)
    original = dataset.records()[0]
    dataset.review(original["id"], status="excluded", boxes=[])
    export_dataset(dataset, output)
    assert not list(output.rglob(f"{original['id']}*"))
    assert dataset.image_path(original).is_file()
    with pytest.raises(ValueError, match="source dataset"):
        export_dataset(dataset, dataset.directory)
    other = tmp_path / "important-folder"
    other.mkdir()
    (other / "notes.txt").write_text("keep me")
    with pytest.raises(ValueError, match="non-export folder"):
        export_dataset(dataset, other)
    assert (other / "notes.txt").read_text() == "keep me"


def test_export_detects_modified_source_and_leaves_previous_export_intact(tmp_path):
    dataset = reviewed_dataset(tmp_path)
    output = tmp_path / "export"
    export_dataset(dataset, output)
    before = (output / "export-report.json").read_bytes()
    dataset.image_path(dataset.records()[0]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed since import"):
        export_dataset(dataset, output)
    assert (output / "export-report.json").read_bytes() == before
