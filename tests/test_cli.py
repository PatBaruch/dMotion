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


@pytest.mark.parametrize("flag", ["--prompt", "--reference", "--engine"])
def test_retired_model_options_are_not_available(flag):
    with pytest.raises(SystemExit):
        parser().parse_args(["run", flag, "unused"])


@pytest.mark.parametrize("arguments", [["--confidence", "1.5"], ["--image-size", "641"]])
def test_invalid_override_fails_before_camera_or_model_load(arguments):
    assert main(["run", *arguments]) == 1


def test_missing_config_produces_failure(tmp_path):
    assert main(["doctor", "--config", str(tmp_path / "missing.toml")]) == 1


def test_selected_checkpoint_reaches_camera_with_user_overrides(monkeypatch):
    received = {}

    def camera(config, *, demo):
        received.update(config=config, demo=demo)
        return 0

    monkeypatch.setitem(
        sys.modules, "dmotion.app", SimpleNamespace(run_camera=camera, run_image=None)
    )
    assert (
        main(
            [
                "run",
                "--mode",
                "trained",
                "--model",
                "candidate.pt",
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
    assert received["demo"] is False
    assert received["config"].detector.model == "candidate.pt"
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

    def camera(config, *, demo):
        received.update(config=config)
        return 0

    monkeypatch.setitem(
        sys.modules, "dmotion.app", SimpleNamespace(run_camera=camera, run_image=None)
    )
    assert main(["run", "--mode", "trained"]) == 0
    assert received["config"].detector.backend == "trained"


def test_trained_mode_confidence_can_be_overridden(monkeypatch):
    received = {}

    def camera(config, *, demo):
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


@pytest.mark.parametrize(
    "passed,require_pass,expected", [(True, True, 0), (False, True, 2), (False, False, 0)]
)
def test_evaluation_cli_routes_explicit_paths_and_reports_gate_status(
    monkeypatch, tmp_path, passed, require_pass, expected, capsys
):
    import json

    received = {}
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "verdict": "passed_frame_gates" if passed else "failed_test",
                "candidate_gate_passed": passed,
            }
        )
    )

    def evaluate(config, directory, split_file, candidate, **kwargs):
        received.update(
            config=config, directory=directory, split_file=split_file, candidate=candidate, **kwargs
        )
        return report_path

    monkeypatch.setitem(
        sys.modules,
        "dmotion.evaluation",
        SimpleNamespace(DEFAULT_THRESHOLDS=(0.5,), evaluate_checkpoints=evaluate),
    )
    arguments = [
        "evaluate",
        "--dataset",
        str(tmp_path),
        "--splits",
        str(tmp_path / "splits.json"),
        "--candidate",
        "candidate.pt",
        "--baseline",
        "baseline.pt",
        "--device",
        "cpu",
        "--image-size",
        "320",
        "--threshold",
        "0.3",
        "--threshold",
        "0.7",
        "--test-context",
        "fresh",
    ]
    assert main(arguments + (["--require-pass"] if require_pass else [])) == expected
    assert received["thresholds"] == [0.3, 0.7]
    assert received["config"].detector.device == "cpu"
    assert received["config"].detector.image_size == 320
    assert received["candidate"].name == "candidate.pt"
    assert received["test_context"] == "fresh"
    assert "Readable report:" in capsys.readouterr().out


def test_evaluation_help_needs_no_vision_packages():
    before = set(sys.modules)
    with pytest.raises(SystemExit) as result:
        main(["evaluate", "--help"])
    assert result.value.code == 0
    assert not ({"torch", "ultralytics", "cv2", "pygame"} & (set(sys.modules) - before))
