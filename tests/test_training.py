import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dmotion.config import AppConfig
from dmotion.dataset import Dataset
from dmotion.training import train_model


def small_dataset(tmp_path):
    dataset = Dataset(tmp_path / "dataset")
    for index in range(3):
        source = tmp_path / f"{index}.jpg"
        source.write_bytes(f"image {index}".encode())
        record = dataset.add_image(source, group=f"shoot-{index}", width=100, height=100)
        dataset.review(record["id"], status="positive", boxes=[[10, 10, 90, 90]])
    return dataset


def fake_dependencies(monkeypatch, *, fail_evaluation=False):
    calls = []

    class FakeYOLO:
        def __init__(self, path):
            calls.append(("load", path))

        def train(self, **arguments):
            calls.append(("train", arguments))
            best = Path(arguments["project"]) / arguments["name"] / "weights" / "best.pt"
            best.parent.mkdir(parents=True)
            best.write_bytes(b"trained checkpoint")
            self.trainer = SimpleNamespace(best=best, epoch=arguments["epochs"] - 1)

        def val(self, **arguments):
            calls.append(("val", arguments))
            if fail_evaluation:
                raise RuntimeError("evaluation failed")
            return SimpleNamespace(results_dict={"metrics/mAP50(B)": 0.42, "invalid": float("nan")})

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "ultralytics",
        SimpleNamespace(
            YOLO=FakeYOLO,
            settings=SimpleNamespace(update=lambda value: None),
        ),
    )
    return calls


def test_training_uses_reviewed_export_and_tests_best_model_before_saving(tmp_path, monkeypatch):
    dataset = small_dataset(tmp_path)
    calls = fake_dependencies(monkeypatch)
    destination = train_model(AppConfig(root=tmp_path), dataset.directory, epochs=2, image_size=320)
    assert destination.read_bytes() == b"trained checkpoint"
    info = json.loads(destination.with_suffix(".json").read_text())
    assert info["test_metrics"] == {"metrics/mAP50(B)": 0.42}
    assert info["completed_epochs"] == 2
    assert info["device"] == "cpu"
    assert info["warnings"]
    assert [event for event, _ in calls] == ["load", "train", "load", "val"]
    training = calls[1][1]
    assert training["workers"] == 0 and training["batch"] == 4 and training["patience"] == 10
    assert training["imgsz"] == 320 and training["pretrained"] is True
    assert calls[3][1]["split"] == "test"
    assert str(Path(info["checkpoint"])) == calls[2][1]
    assert (tmp_path / "outputs" / "training" / info["run"] / "training-info.json").is_file()
    provenance = tmp_path / "outputs" / "training" / info["run"] / "dataset-provenance.json"
    assert len(json.loads(provenance.read_text())["records"]) == 3


def test_evaluation_failure_does_not_replace_previous_model(tmp_path, monkeypatch):
    dataset = small_dataset(tmp_path)
    fake_dependencies(monkeypatch, fail_evaluation=True)
    destination = tmp_path / "models" / "money-spread.pt"
    destination.parent.mkdir()
    destination.write_bytes(b"previous model")
    with pytest.raises(RuntimeError, match="evaluation failed"):
        train_model(AppConfig(root=tmp_path), dataset.directory, epochs=1)
    assert destination.read_bytes() == b"previous model"


@pytest.mark.parametrize(("epochs", "image_size"), [(0, 640), (501, 640), (True, 640), (1, 410)])
def test_bad_training_arguments_fail_before_export_or_model_import(tmp_path, epochs, image_size):
    with pytest.raises(ValueError):
        train_model(
            AppConfig(root=tmp_path), tmp_path / "dataset", epochs=epochs, image_size=image_size
        )
    assert not (tmp_path / "dataset").exists()
