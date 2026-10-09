import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dmotion.config import AppConfig
from dmotion.dataset import Dataset, export_dataset
from dmotion.training import train_model


def small_dataset(tmp_path):
    dataset = Dataset(tmp_path / "dataset")
    for index in range(3):
        source = tmp_path / f"{index}.jpg"
        source.write_bytes(f"image {index}".encode())
        record = dataset.add_image(source, group=f"shoot-{index}", width=100, height=100)
        dataset.review(record["id"], status="positive", boxes=[[10, 10, 90, 90]])
    return dataset


def fake_dependencies(
    monkeypatch, *, fail_evaluation=False, optimizer_lrs=None, on_load=None, on_train=None
):
    calls = []

    class FakeOptimizer:
        def __init__(self):
            self.param_groups = [{"lr": 0.002}]
            self.hooks = []

        def register_step_post_hook(self, callback):
            self.hooks.append(callback)
            return SimpleNamespace(remove=lambda: self.hooks.remove(callback))

        def step(self, learning_rate):
            self.param_groups[0]["lr"] = learning_rate
            for hook in self.hooks:
                hook(self, (), {})

    class FakeYOLO:
        def __init__(self, path):
            calls.append(("load", path))
            initial = Path(path)
            initial.parent.mkdir(parents=True, exist_ok=True)
            if not initial.exists():
                initial.write_bytes(b"generic YOLO26m initialization")
            if on_load:
                on_load(initial)
            self.callbacks = {}

        def add_callback(self, event, callback):
            self.callbacks.setdefault(event, []).append(callback)

        def train(self, **arguments):
            calls.append(("train", arguments))
            if on_train:
                on_train()
            best = Path(arguments["project"]) / arguments["name"] / "weights" / "best.pt"
            best.parent.mkdir(parents=True)
            best.write_bytes(b"trained checkpoint")
            self.trainer = SimpleNamespace(
                best=best, epoch=arguments["epochs"] - 1, optimizer=FakeOptimizer()
            )
            for callback in self.callbacks.get("on_train_start", []):
                callback(self.trainer)
            rates = optimizer_lrs if optimizer_lrs is not None else [0.002] * arguments["epochs"]
            for learning_rate in rates:
                self.trainer.optimizer.step(learning_rate)

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


def test_training_uses_yolo26m_and_saves_candidate_without_evaluating_test(tmp_path, monkeypatch):
    dataset = small_dataset(tmp_path)
    calls = fake_dependencies(monkeypatch)
    destination = train_model(AppConfig(root=tmp_path), dataset.directory, epochs=2, image_size=320)
    assert destination.read_bytes() == b"trained checkpoint"
    info = json.loads(destination.with_suffix(".json").read_text())
    assert info["test_evaluated"] is False
    assert info["model_id"] == "yolo26m.pt"
    assert info["model_revision"] == f"sha256:{info['pretrained_sha256']}"
    assert info["active_model_replaced"] is False
    assert calls[0][1] == str(tmp_path / "models/yolo26m.pt")
    assert info["completed_epochs"] == 2
    assert info["patience"] == 10
    assert info["optimizer_steps"] == info["positive_lr_optimizer_steps"] == 2
    assert info["device"] == "cpu"
    assert info["warnings"]
    assert [event for event, _ in calls] == ["load", "load", "train"]
    training = calls[2][1]
    assert training["workers"] == 0 and training["batch"] == 2 and training["patience"] == 10
    assert training["nbs"] == training["batch"]
    assert training["imgsz"] == 320 and training["pretrained"] is True
    assert (
        tmp_path / "outputs" / "yolo26m-training" / info["run"] / "training-info.json"
    ).is_file()
    provenance = tmp_path / "outputs" / "yolo26m-training" / info["run"] / "dataset-provenance.json"
    assert len(json.loads(provenance.read_text())["records"]) == 3


def test_successful_training_preserves_live_checkpoint_and_report(tmp_path, monkeypatch):
    dataset = small_dataset(tmp_path)
    calls = fake_dependencies(monkeypatch)
    active = tmp_path / "models/cash-yolo26m.pt"
    active.parent.mkdir()
    active.write_bytes(b"active cash model")
    report = active.with_suffix(".json")
    report.write_bytes(b"existing model report")
    candidate = train_model(AppConfig(root=tmp_path), dataset.directory, epochs=1)
    assert candidate != active
    assert active.read_bytes() == b"active cash model"
    assert report.read_bytes() == b"existing model report"
    assert candidate.read_bytes() == b"trained checkpoint"
    assert not any(event == "val" for event, _ in calls)


def test_pending_ai_labels_fail_before_export_or_heavy_imports(tmp_path, monkeypatch):
    dataset = small_dataset(tmp_path)
    export_directory = tmp_path / "data" / "yolo"
    export_dataset(dataset, export_directory)
    existing_export = {
        path.relative_to(export_directory): path.read_bytes()
        for path in export_directory.rglob("*")
        if path.is_file()
    }
    destination = tmp_path / "models" / "cash-yolo26m.pt"
    destination.parent.mkdir()
    destination.write_bytes(b"previous model")
    metadata = destination.with_suffix(".json")
    metadata.write_text('{"previous": true}')
    for index, boxes in enumerate(([[10, 10, 90, 90]], [])):
        source = tmp_path / f"pending-{index}.jpg"
        source.write_bytes(f"pending image {index}".encode())
        record = dataset.add_image(source, group="new-video", width=100, height=100)
        dataset.suggest(
            record["id"],
            {
                "boxes": boxes,
                "scores": [0.8] * len(boxes),
                "labels": ["cash money"] * len(boxes),
                "model": "grounding-dino-tiny",
                "prompts": ["cash money"],
                "created_at": "2026-10-02T21:54:27+00:00",
            },
        )
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    with pytest.raises(ValueError, match="2 AI-labeled pictures still need review.*make label"):
        train_model(AppConfig(root=tmp_path), dataset.directory)
    assert destination.read_bytes() == b"previous model"
    assert json.loads(metadata.read_text()) == {"previous": True}
    assert existing_export == {
        path.relative_to(export_directory): path.read_bytes()
        for path in export_directory.rglob("*")
        if path.is_file()
    }
    assert not (tmp_path / ".cache").exists()


def test_unreviewed_pictures_without_ai_drafts_do_not_block_training(tmp_path, monkeypatch):
    dataset = small_dataset(tmp_path)
    source = tmp_path / "later.jpg"
    source.write_bytes(b"future unreviewed image")
    pending = dataset.add_image(source, group="later-session", width=100, height=100)
    fake_dependencies(monkeypatch)
    destination = train_model(AppConfig(root=tmp_path), dataset.directory, epochs=1)
    info = json.loads(destination.with_suffix(".json").read_text())
    provenance = json.loads(
        Path(info["checkpoint"]).parent.parent.joinpath("dataset-provenance.json").read_text()
    )
    assert len(provenance["records"]) == 3
    assert pending["id"] not in {record["id"] for record in provenance["records"]}


def test_counts_actual_optimizer_steps_instead_of_epochs(tmp_path, monkeypatch):
    dataset = small_dataset(tmp_path)
    fake_dependencies(monkeypatch, optimizer_lrs=[0.0, 0.002, float("nan")])
    destination = train_model(AppConfig(root=tmp_path), dataset.directory, epochs=10)
    info = json.loads(destination.with_suffix(".json").read_text())
    assert info["completed_epochs"] == 10
    assert info["optimizer_steps"] == 3
    assert info["positive_lr_optimizer_steps"] == 1


@pytest.mark.parametrize("patience", [0, 30])
def test_explicit_patience_reaches_training_and_metadata(tmp_path, monkeypatch, patience):
    dataset = small_dataset(tmp_path)
    calls = fake_dependencies(monkeypatch)
    destination = train_model(
        AppConfig(root=tmp_path), dataset.directory, epochs=1, patience=patience
    )
    assert calls[2][1]["patience"] == patience
    assert json.loads(destination.with_suffix(".json").read_text())["patience"] == patience


@pytest.mark.parametrize("patience", [-1, 501, True, 1.5, "30"])
def test_bad_patience_fails_before_dataset_creation_or_model_import(tmp_path, patience):
    with pytest.raises(ValueError, match="patience must be an integer between 0 and 500"):
        train_model(AppConfig(root=tmp_path), tmp_path / "dataset", patience=patience)
    assert not (tmp_path / "dataset").exists()


@pytest.mark.parametrize("optimizer_lrs", [[], [0.0], [float("nan")]])
def test_no_learning_updates_preserves_existing_model(tmp_path, monkeypatch, optimizer_lrs):
    dataset = small_dataset(tmp_path)
    calls = fake_dependencies(monkeypatch, optimizer_lrs=optimizer_lrs)
    destination = tmp_path / "models" / "cash-yolo26m.pt"
    destination.parent.mkdir()
    destination.write_bytes(b"previous model")
    metadata = destination.with_suffix(".json")
    metadata.write_text('{"previous": true}')
    with pytest.raises(RuntimeError, match="no optimizer updates at a positive learning rate"):
        train_model(AppConfig(root=tmp_path), dataset.directory, epochs=1)
    assert destination.read_bytes() == b"previous model"
    assert json.loads(metadata.read_text()) == {"previous": True}
    assert not any(event == "val" for event, _ in calls)


@pytest.mark.parametrize(("epochs", "image_size"), [(0, 640), (501, 640), (True, 640), (1, 410)])
def test_bad_training_arguments_fail_before_export_or_model_import(tmp_path, epochs, image_size):
    with pytest.raises(ValueError):
        train_model(
            AppConfig(root=tmp_path), tmp_path / "dataset", epochs=epochs, image_size=image_size
        )
    assert not (tmp_path / "dataset").exists()


@pytest.mark.parametrize("batch", [-1, 0, True, 1.5, 65])
def test_invalid_batch_fails_before_creating_dataset(tmp_path, batch):
    with pytest.raises(ValueError, match="batch must be an integer"):
        train_model(AppConfig(root=tmp_path), tmp_path / "dataset", batch=batch)
    assert not (tmp_path / "dataset").exists()


def test_smaller_batch_reaches_optimizer_accumulation_and_report(tmp_path, monkeypatch):
    dataset = small_dataset(tmp_path)
    calls = fake_dependencies(monkeypatch)
    candidate = train_model(AppConfig(root=tmp_path), dataset.directory, batch=1)
    assert calls[2][1]["batch"] == calls[2][1]["nbs"] == 1
    assert json.loads(candidate.with_suffix(".json").read_text())["batch"] == 1


def test_training_provenance_hashes_loaded_snapshot_and_keeps_it_for_train(tmp_path, monkeypatch):
    dataset = small_dataset(tmp_path)
    original = tmp_path / "models/yolo26m.pt"
    original.parent.mkdir()
    original.write_bytes(b"original pretrained weights")
    loaded = []

    def replace_original(snapshot):
        replacement = original.with_name("replacement.pt")
        replacement.write_bytes(b"replacement pretrained weights")
        replacement.replace(original)
        loaded.append((snapshot, snapshot.read_bytes()))

    def check_snapshot_during_train():
        assert loaded[0][0].read_bytes() == b"original pretrained weights"

    calls = fake_dependencies(
        monkeypatch, on_load=replace_original, on_train=check_snapshot_during_train
    )
    candidate = train_model(AppConfig(root=tmp_path), dataset.directory, epochs=1)
    info = json.loads(candidate.with_suffix(".json").read_text())
    assert info["pretrained"] == str(original)
    assert info["pretrained_sha256"] == hashlib.sha256(loaded[0][1]).hexdigest()
    assert info["model_revision"] == "sha256:" + info["pretrained_sha256"]
    assert original.read_bytes() == b"replacement pretrained weights"
    assert [event for event, _ in calls] == ["load", "train"]
    assert not loaded[0][0].exists()
