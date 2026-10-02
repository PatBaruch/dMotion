from dataclasses import replace
from pathlib import Path

import pytest

from dmotion.config import AppConfig, load_config, validate


def test_repository_config_loads_and_resolves_paths():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "config.toml")
    assert config.resolve(config.detector.model) == root / "models/yolov8s-worldv2.pt"
    assert "banknotes" in config.detector.prompts


def test_paths_resolve_against_config_file_not_working_directory(tmp_path, monkeypatch):
    settings = tmp_path / "settings"
    settings.mkdir()
    config_path = settings / "local.toml"
    config_path.write_text('[audio]\nfile = "sound.wav"\n')
    monkeypatch.chdir(tmp_path)
    config = load_config(config_path)
    assert config.resolve(config.audio.file) == settings / "sound.wav"


@pytest.mark.parametrize("value", [-0.1, 1.1, "high", True, float("nan")])
def test_rejects_invalid_confidence(value, tmp_path):
    config = AppConfig(root=tmp_path)
    config = replace(config, detector=replace(config.detector, confidence=value))
    with pytest.raises(ValueError, match="confidence"):
        validate(config)


@pytest.mark.parametrize(
    "text",
    [
        "[detector]\nprompts = []",
        '[detector]\nprompts = "cash"',
        '[detector]\nprompts = [""]',
        "[detector]\nimage_size = 410",
        "[detector]\nconfdence = 0.5",
        "[trigger]\nconsecutive_hits = 0",
        '[camera]\nmirror = "false"',
        "[unknown]\nvalue = 1",
    ],
)
def test_bad_settings_fail_before_loading_a_model(text, tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_config(path)
