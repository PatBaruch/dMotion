import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from dmotion.config import AppConfig
from dmotion.detector import MoneyDetector


def install_fake_models(monkeypatch, *, names=None):
    calls = []

    class FakeModel:
        def __init__(self, path, backend):
            calls.append(("load", backend, path))
            self.names = {0: "money_spread"} if names is None else names

        def set_classes(self, prompts):
            calls.append(("set_classes", prompts))

        def to(self, device):
            calls.append(("to", device))

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
            YOLO=lambda path: FakeModel(path, "trained"),
            YOLOWorld=lambda path: FakeModel(path, "world"),
            settings=SimpleNamespace(update=lambda value: None),
        ),
    )
    return calls


def trained_config(tmp_path):
    config = AppConfig(root=tmp_path)
    return replace(
        config,
        detector=replace(
            config.detector,
            backend="trained",
            model="models/money-spread.pt",
        ),
    )


@pytest.mark.parametrize("names", [{0: "money_spread"}, {0: "cash"}])
def test_trained_detector_loads_custom_yolo_without_overwriting_learned_class_names(
    tmp_path,
    monkeypatch,
    names,
):
    calls = install_fake_models(monkeypatch, names=names)
    config = trained_config(tmp_path)
    path = config.resolve(config.detector.model)
    path.parent.mkdir()
    path.write_bytes(b"model placeholder")
    detector = MoneyDetector(config)
    assert detector.device == "cpu"
    assert calls == [("load", "trained", str(path)), ("to", "cpu")]


@pytest.mark.parametrize("names", [{}, {0: "person"}, {0: "money_spread", 1: "person"}])
def test_trained_detector_rejects_other_detection_classes(tmp_path, monkeypatch, names):
    calls = install_fake_models(monkeypatch, names=names)
    config = trained_config(tmp_path)
    path = config.resolve(config.detector.model)
    path.parent.mkdir()
    path.write_bytes(b"model placeholder")
    with pytest.raises(ValueError, match="money_spread or cash class"):
        MoneyDetector(config)
    assert not any(call[0] == "to" for call in calls)


def test_trained_detector_explains_missing_checkpoint_without_downloading_a_generic_model(
    tmp_path,
    monkeypatch,
):
    calls = install_fake_models(monkeypatch)
    with pytest.raises(FileNotFoundError, match="Review your photos and run make train"):
        MoneyDetector(trained_config(tmp_path))
    assert calls == []


def test_world_detector_still_encodes_requested_prompts(tmp_path, monkeypatch):
    calls = install_fake_models(monkeypatch)
    config = AppConfig(root=tmp_path)
    MoneyDetector(config)
    assert calls[0][1] == "world"
    assert calls[1] == ("set_classes", list(config.detector.prompts))
    assert calls[2] == ("to", "cpu")
