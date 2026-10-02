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


def test_invalid_override_fails_before_camera_or_model_load(capsys):
    assert main(["run", "--confidence", "1.5"]) == 1


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
    assert received["config"].camera.index == 1
    assert received["config"].audio.enabled is False
