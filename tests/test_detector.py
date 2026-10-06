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
            model="models/cash-yolo26m.pt",
        ),
    )


def test_trained_detector_loads_custom_yolo_without_overwriting_learned_class_names(
    tmp_path,
    monkeypatch,
):
    calls = install_fake_models(monkeypatch)
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
    with pytest.raises(ValueError, match="single cash or money_spread class"):
        MoneyDetector(config)
    assert not any(call[0] == "to" for call in calls)


def test_trained_detector_explains_missing_checkpoint_without_downloading_a_generic_model(
    tmp_path,
    monkeypatch,
):
    calls = install_fake_models(monkeypatch)
    with pytest.raises(FileNotFoundError, match="Supply --model"):
        MoneyDetector(trained_config(tmp_path))
    assert calls == []


def test_semantic_cash_class_is_accepted_without_prompt_encoding(tmp_path, monkeypatch):
    calls = install_fake_models(monkeypatch, names={0: "cash"})
    config = trained_config(tmp_path)
    path = config.resolve(config.detector.model)
    path.parent.mkdir()
    path.write_bytes(b"cash checkpoint")
    MoneyDetector(config)
    assert calls == [("load", "trained", str(path)), ("to", "cpu")]
