import hashlib
import sys
from dataclasses import replace
from pathlib import Path
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
    loaded = Path(calls[0][2])
    assert loaded.name == path.name and loaded != path and not loaded.exists()
    assert calls[1] == ("to", "cpu")
    assert detector.model_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


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
    assert Path(calls[0][2]).name == path.name
    assert calls[1] == ("to", "cpu")


def test_detector_hash_matches_loaded_bytes_when_original_is_replaced(tmp_path, monkeypatch):
    install_fake_models(monkeypatch)
    config = trained_config(tmp_path)
    original = config.resolve(config.detector.model)
    original.parent.mkdir()
    original.write_bytes(b"old cash checkpoint")
    loaded = []

    class Model:
        names = {0: "cash"}

        def __init__(self, path):
            replacement = original.with_name("replacement.pt")
            replacement.write_bytes(b"new cash checkpoint")
            replacement.replace(original)
            loaded.append(Path(path).read_bytes())

        def to(self, device):
            pass

    monkeypatch.setattr(sys.modules["ultralytics"], "YOLO", Model)
    detector = MoneyDetector(config)
    assert loaded == [b"old cash checkpoint"]
    assert original.read_bytes() == b"new cash checkpoint"
    assert detector.model_sha256 == hashlib.sha256(loaded[0]).hexdigest()
    assert list((tmp_path / ".cache/checkpoints").iterdir()) == []
