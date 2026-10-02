from dataclasses import replace

import pytest

from dmotion.config import AppConfig, apply_mode


def test_money_mode_preserves_original_settings(tmp_path):
    config = AppConfig(root=tmp_path)
    assert apply_mode(config, "money") is config


def test_check_mode_detects_common_objects_and_confirms_first_match(tmp_path):
    config = AppConfig(root=tmp_path)
    checking = apply_mode(config, "check")
    assert checking.detector.prompts == ("person", "cell phone", "cup", "bottle", "book")
    assert checking.detector.confidence == 0.25
    assert checking.trigger.consecutive_hits == 1


def test_check_mode_preserves_camera_audio_and_other_detector_trigger_settings(tmp_path):
    config = AppConfig(root=tmp_path)
    config = replace(
        config,
        detector=replace(config.detector, model="custom.pt", image_size=640, device="cpu"),
        camera=replace(config.camera, index=2, mirror=False, max_result_age_seconds=4.0),
        audio=replace(config.audio, enabled=False, file="alert.wav", volume=0.3),
        trigger=replace(config.trigger, reset_seconds=2.0, cooldown_seconds=5.0),
        overlay=replace(config.overlay, solid_box=True),
    )
    checking = apply_mode(config, "check")
    assert checking.root == config.root
    assert checking.camera == config.camera
    assert checking.audio == config.audio
    assert checking.overlay == config.overlay
    assert checking.detector.model == "custom.pt"
    assert checking.detector.image_size == 640
    assert checking.detector.device == "cpu"
    assert checking.trigger.reset_seconds == 2.0
    assert checking.trigger.cooldown_seconds == 5.0


def test_check_mode_does_not_mutate_money_prompts_or_sensitivity(tmp_path):
    config = AppConfig(root=tmp_path)
    config = replace(
        config,
        detector=replace(config.detector, prompts=("banknote",), confidence=0.1),
        trigger=replace(config.trigger, consecutive_hits=5),
    )
    checking = apply_mode(config, "check")
    assert checking.detector.confidence == 0.25
    assert checking.trigger.consecutive_hits == 1
    assert config.detector.prompts == ("banknote",)
    assert config.detector.confidence == 0.1
    assert config.trigger.consecutive_hits == 5


def test_unsupported_mode_fails_explicitly(tmp_path):
    with pytest.raises(ValueError):
        apply_mode(AppConfig(root=tmp_path), "unknown")
