import sys

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
