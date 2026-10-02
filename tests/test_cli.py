import sys
from types import SimpleNamespace

import pytest

from dmotion.cli import main, parser


def test_help_does_not_import_vision_dependencies():
    before = set(sys.modules)
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert not ({"torch", "ultralytics", "cv2", "pygame"} & (set(sys.modules) - before))


def test_repeated_prompts_are_supported():
    args = parser().parse_args(["run", "--prompt", "banknotes", "--prompt", "cash"])
    assert args.prompt == ["banknotes", "cash"]


@pytest.mark.parametrize("arguments", [["--confidence", "1.5"], ["--image-size", "641"]])
def test_invalid_override_fails_before_camera_or_model_load(arguments):
    assert main(["run", *arguments]) == 1


def test_missing_config_produces_failure(tmp_path):
    assert main(["doctor", "--config", str(tmp_path / "missing.toml")]) == 1


def test_check_mode_reaches_camera_with_user_overrides(monkeypatch):
    received = {}

    def camera(config, *, demo, checking):
        received.update(config=config, demo=demo, checking=checking)
        return 0

    monkeypatch.setitem(
        sys.modules, "dmotion.app", SimpleNamespace(run_camera=camera, run_image=None)
    )
    assert (
        main(
            [
                "run",
                "--mode",
                "check",
                "--confidence",
                "0.1",
                "--image-size",
                "640",
                "--camera",
                "1",
                "--mute",
            ]
        )
        == 0
    )
    assert received["checking"] is True
    assert received["demo"] is False
    assert "person" in received["config"].detector.prompts
    assert received["config"].detector.confidence == 0.1
    assert received["config"].detector.image_size == 640
    assert received["config"].camera.index == 1
    assert received["config"].audio.enabled is False


def test_collect_success_returns_zero_even_when_multiple_frames_saved(monkeypatch, tmp_path):
    received = {}

    def collect(config, directory, **kwargs):
        received.update(config=config, directory=directory, **kwargs)
        return 20

    monkeypatch.setitem(sys.modules, "dmotion.collect", SimpleNamespace(record_camera=collect))
    assert main(["collect", "--dataset", str(tmp_path), "--kind", "negative"]) == 0
    assert received["kind"] == "negative"
    assert received["directory"] == tmp_path


def test_trained_mode_uses_custom_backend(monkeypatch):
    received = {}

    def camera(config, *, demo, checking):
        received.update(config=config, checking=checking)
        return 0

    monkeypatch.setitem(
        sys.modules, "dmotion.app", SimpleNamespace(run_camera=camera, run_image=None)
    )
    assert main(["run", "--mode", "trained"]) == 0
    assert received["config"].detector.backend == "trained"
    assert received["checking"] is False


def test_trained_mode_confidence_can_be_overridden(monkeypatch):
    received = {}

    def camera(config, *, demo, checking):
        received.update(config=config)
        return 0

    monkeypatch.setitem(
        sys.modules, "dmotion.app", SimpleNamespace(run_camera=camera, run_image=None)
    )
    assert main(["run", "--mode", "trained", "--confidence", "0.2"]) == 0
    assert received["config"].detector.confidence == 0.2


@pytest.mark.parametrize(("arguments", "expected"), [([], 10), (["--patience", "30"], 30)])
def test_training_patience_reaches_training_function(monkeypatch, tmp_path, arguments, expected):
    received = {}

    def train(config, directory, **kwargs):
        received.update(kwargs)
        return tmp_path / "trained.pt"

    monkeypatch.setitem(sys.modules, "dmotion.training", SimpleNamespace(train_model=train))
    assert main(["train", "--dataset", str(tmp_path), *arguments]) == 0
    assert received["patience"] == expected
