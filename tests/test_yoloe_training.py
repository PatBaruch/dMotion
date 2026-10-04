import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dmotion.config import AppConfig
from dmotion.dataset import Dataset, export_dataset
from dmotion.yoloe_training import head_freeze, snapshot_reviewed, train_yoloe


def reviewed(tmp_path, name="source"):
    dataset = Dataset(tmp_path / name)
    for index in range(3):
        for status in ("positive", "negative"):
            image = tmp_path / f"{index}-{status}.jpg"
            image.write_bytes(f"frame-{index}-{status}".encode())
            row = dataset.add_image(image, group=f"group-{index}", width=100, height=100)
            dataset.review(
                row["id"], status=status, boxes=[[10, 10, 50, 50]] if status == "positive" else []
            )
    return dataset


def checkpoint(tmp_path):
    path = tmp_path / "models/yoloe-26s-seg.pt"
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(b"selected pretrained checkpoint")
    return path


def fake_ml(
    monkeypatch,
    *,
    invalid_trainable=False,
    update_lr=0.001,
    fail_validation=False,
    missing_checkpoint=False,
):
    calls = []

    class Head:
        def named_children(self):
            return [(name, object()) for name in ("cv2", "cv3", "one2one_cv3", "dfl", "savpe")]

    class Optimizer:
        param_groups = [{"lr": update_lr}]

        def register_step_post_hook(self, fn):
            self.fn = fn
            return SimpleNamespace(remove=lambda: calls.append(("remove-hook", None)))

    class FakeYOLOE:
        def __init__(self, architecture):
            calls.append(("architecture", architecture))
            self.model = SimpleNamespace(model=[object(), Head()])

        def load(self, weights):
            calls.append(("weights", weights))
            return self

        def add_callback(self, event, fn):
            assert event == "on_train_start"
            self.callback = fn

        def train(self, **kwargs):
            calls.append(("train", kwargs))
            best = Path(kwargs["project"]) / kwargs["name"] / "weights/best.pt"
            best.parent.mkdir(parents=True)
            if not missing_checkpoint:
                best.write_bytes(b"candidate checkpoint")
            parameter = SimpleNamespace(requires_grad=True, numel=lambda: 42)
            key = "model.0.weight" if invalid_trainable else "model.1.cv3.0.2.weight"
            model = SimpleNamespace(named_parameters=lambda: [(key, parameter)])
            optimizer = Optimizer()
            self.trainer = SimpleNamespace(
                model=model, optimizer=optimizer, best=best, epoch=kwargs["epochs"] - 1
            )
            self.callback(self.trainer)
            for _ in range(kwargs["epochs"]):
                optimizer.fn(optimizer, (), {})

    class FakeYOLO:
        def __init__(self, weights):
            calls.append(("evaluate-weights", weights))

        def val(self, **kwargs):
            calls.append(("validation", kwargs))
            if fail_validation:
                raise RuntimeError("validation failed")
            return SimpleNamespace(results_dict={"precision": 0.8})

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
            YOLO=FakeYOLO, YOLOE=FakeYOLOE, settings=SimpleNamespace(update=lambda _: None)
        ),
    )
    trainer = object()
    monkeypatch.setitem(
        sys.modules, "ultralytics.models.yolo.yoloe", SimpleNamespace(YOLOEPETrainer=trainer)
    )
    monkeypatch.setattr("dmotion.yoloe_training.version", lambda _: "8.4.171")
    return calls, trainer


def test_snapshot_deduplicates_preserves_reviews_groups_and_origins(tmp_path):
    first = reviewed(tmp_path)
    second = Dataset(tmp_path / "second")
    for row in first.records():
        copy = second.add_image(first.image_path(row), group=row["group"], width=100, height=100)
        second.review(copy["id"], status=row["status"], boxes=row["boxes"])
    before = first.manifest_path.read_bytes()
    snapshot = snapshot_reviewed([first.directory, second.directory], tmp_path / "snapshot")
    assert len(snapshot.records()) == 6
    assert all(len(row["origins"]) == 2 for row in snapshot.records())
    assert {r["group"] for r in snapshot.records()} == {"group-0", "group-1", "group-2"}
    assert snapshot.summary()["positive"] == snapshot.summary()["negative"] == 3
    assert first.manifest_path.read_bytes() == before


@pytest.mark.parametrize("conflict", ["boxes", "group", "status"])
def test_conflicting_duplicate_reviews_fail_without_snapshot(tmp_path, conflict):
    first = reviewed(tmp_path)
    second = Dataset(tmp_path / "second")
    row = first.records()[0]
    group = "different" if conflict == "group" else row["group"]
    copy = second.add_image(first.image_path(row), group=group, width=100, height=100)
    second.review(
        copy["id"],
        status="negative" if conflict == "status" else "positive",
        boxes=[]
        if conflict == "status"
        else [[11, 11, 50, 50]]
        if conflict == "boxes"
        else row["boxes"],
    )
    with pytest.raises(ValueError, match="Conflicting"):
        snapshot_reviewed([first.directory, second.directory], tmp_path / "snapshot")
    assert not (tmp_path / "snapshot").exists()


def test_pending_review_is_not_a_negative_and_changed_images_are_rejected(tmp_path):
    source = reviewed(tmp_path)
    image = tmp_path / "pending.jpg"
    image.write_bytes(b"new frame")
    row = source.add_image(image, group="new", width=100, height=100)
    with pytest.raises(ValueError, match="unreviewed"):
        snapshot_reviewed([source.directory], tmp_path / "snapshot")
    source.review(row["id"], status="excluded", boxes=[])
    source.image_path(source.records()[0]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed since review"):
        snapshot_reviewed([source.directory], tmp_path / "snapshot")


def test_snapshot_requires_real_input_and_new_output(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        snapshot_reviewed([], tmp_path / "out")
    with pytest.raises(ValueError, match="manifest is missing"):
        snapshot_reviewed([tmp_path / "missing"], tmp_path / "out")
    empty = Dataset(tmp_path / "empty")
    empty.manifest_path.write_text('{"version":1,"records":[]}')
    with pytest.raises(ValueError, match="No reviewed"):
        snapshot_reviewed([empty.directory], tmp_path / "out")
    source = reviewed(tmp_path)
    with pytest.raises(ValueError, match="must be new"):
        snapshot_reviewed([source.directory], source.directory)


def test_head_freeze_keeps_only_terminal_classification_branches():
    head = SimpleNamespace(
        named_children=lambda: [(n, None) for n in ("cv2", "cv3", "one2one_cv3", "reprta")]
    )
    freeze, allowed = head_freeze(SimpleNamespace(model=SimpleNamespace(model=[None, head])))
    assert "0" in freeze and "1.cv2" in freeze and "1.reprta" in freeze
    assert "1.cv3.0.0" in freeze and "1.one2one_cv3.2.1" in freeze
    assert "1.cv3.0.2" not in freeze and "one2one_" in allowed


def test_training_is_candidate_only_audited_and_defers_test_evaluation(tmp_path, monkeypatch):
    source = reviewed(tmp_path)
    checkpoint(tmp_path)
    active = tmp_path / "models/money-spread.pt"
    active.write_bytes(b"working live model")
    reference = (tmp_path / "models/yoloe-26s-seg.pt").read_bytes()
    calls, trainer = fake_ml(monkeypatch)
    candidate = train_yoloe(AppConfig(root=tmp_path), [source.directory], epochs=2)
    info = json.loads(candidate.with_suffix(".json").read_text())
    assert candidate.read_bytes() == b"candidate checkpoint"
    assert active.read_bytes() == b"working live model"
    assert (tmp_path / "models/yoloe-26s-seg.pt").read_bytes() == reference
    assert info["stage"] == "complete" and info["class"] == "cash"
    assert info["model_revision"].startswith("sha256:")
    assert info["trainable_parameter_count"] == 42
    assert info["positive_lr_optimizer_steps"] == info["optimizer_steps"] == 2
    assert info["ultralytics_version"] == "8.4.171" and info["candidate_only"]
    train = next(value for key, value in calls if key == "train")
    assert train["trainer"] is trainer and train["optimizer"] == "AdamW"
    assert train["batch"] == train["nbs"] == 4 and not train["amp"]
    assert train["mosaic"] == 0 and "1.cv2" in train["freeze"]
    val = [v for k, v in calls if k == "validation"]
    assert len(val) == 1 and val[0]["split"] == "val"
    assert calls[-1][0] == "remove-hook"


@pytest.mark.parametrize(
    "options,message",
    [
        ({"invalid_trainable": True}, "unexpected trainable"),
        ({"update_lr": 0}, "No positive-learning-rate"),
        ({"missing_checkpoint": True}, "without a best checkpoint"),
        ({"fail_validation": True}, "validation failed"),
    ],
)
def test_failed_training_never_saves_candidate(tmp_path, monkeypatch, options, message):
    source = reviewed(tmp_path)
    checkpoint(tmp_path)
    fake_ml(monkeypatch, **options)
    with pytest.raises(RuntimeError, match=message):
        train_yoloe(AppConfig(root=tmp_path), [source.directory], epochs=1)
    assert not list((tmp_path / "outputs/yoloe-training").rglob("candidate.pt"))
    report = next((tmp_path / "outputs/yoloe-training").rglob("training-info.json"))
    assert json.loads(report.read_text())["stage"] == "failed"


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"epochs": 0}, "Epochs"),
        ({"patience": -1}, "Patience"),
        ({"image_size": 33}, "Image size"),
        ({"device": ""}, "Device"),
    ],
)
def test_bad_arguments_fail_before_creating_outputs(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        train_yoloe(AppConfig(root=tmp_path), [], **kwargs)
    assert not (tmp_path / "outputs").exists()


def test_checkpoint_must_exist_and_match_selected_model(tmp_path):
    with pytest.raises(ValueError, match="Prepare"):
        train_yoloe(AppConfig(root=tmp_path), [])
    other = tmp_path / "other.pt"
    other.write_bytes(b"other architecture")
    with pytest.raises(ValueError, match="selected"):
        train_yoloe(AppConfig(root=tmp_path), [], pretrained=other)


def test_explicit_split_plan_is_preserved_and_cash_semantics_exported(tmp_path):
    source = reviewed(tmp_path)
    splits = {"group-0": "train", "group-1": "val", "group-2": "test"}
    data = export_dataset(source, tmp_path / "export", class_name="cash", group_splits=splits)
    assert "0: cash" in data.read_text()
    report = json.loads((data.parent / "export-report.json").read_text())
    assert report["group_splits"] == splits and report["class"] == "cash"
    for row in report["records"]:
        assert row["split"] == splits[row["group"]]


@pytest.mark.parametrize(
    "splits",
    [
        {},
        {"group-0": "train", "group-1": "other", "group-2": "test"},
        {"group-0": "train", "group-1": "train", "group-2": "train"},
    ],
)
def test_invalid_split_plans_fail_before_replacing_export(tmp_path, splits):
    source = reviewed(tmp_path)
    with pytest.raises(ValueError):
        export_dataset(source, tmp_path / "export", group_splits=splits)
    assert not (tmp_path / "export").exists()


def test_yoloe_cli_routes_multiple_datasets_and_candidate_output(tmp_path, monkeypatch, capsys):
    from dmotion.cli import main

    config = tmp_path / "config.toml"
    config.write_text("")
    calls = []

    def train(config, datasets, **kwargs):
        calls.append((config, datasets, kwargs))
        return tmp_path / "candidate.pt"

    monkeypatch.setattr("dmotion.yoloe_training.train_yoloe", train)
    assert (
        main(
            [
                "train-yoloe",
                "--config",
                str(config),
                "--dataset",
                "first",
                "second",
                "--output",
                "runs",
                "--pretrained",
                "models/yoloe-26s-seg.pt",
                "--split-file",
                "splits.json",
            ]
        )
        == 0
    )
    assert calls[0][1] == [tmp_path / "first", tmp_path / "second"]
    assert calls[0][2]["output"] == tmp_path / "runs"
    assert calls[0][2]["patience"] == 8
    assert "Calibrate on validation" in capsys.readouterr().out
