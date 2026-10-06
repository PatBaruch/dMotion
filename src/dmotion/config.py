"""Typed, validated TOML settings; independent of camera/ML dependencies."""

import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DetectorConfig:
    model: str = "models/cash-yolo26m.pt"
    backend: str = "trained"
    confidence: float = 0.55
    image_size: int = 640
    device: str = "auto"


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    mirror: bool = True
    max_result_age_seconds: float = 2.0


@dataclass(frozen=True)
class TriggerConfig:
    consecutive_hits: int = 3
    reset_seconds: float = 1.0
    cooldown_seconds: float = 3.0


@dataclass(frozen=True)
class AudioConfig:
    enabled: bool = True
    file: str = "assets/motion_detected.wav"
    volume: float = 0.7


@dataclass(frozen=True)
class OverlayConfig:
    solid_box: bool = False


@dataclass(frozen=True)
class AppConfig:
    root: Path
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)

    def resolve(self, path: str) -> Path:
        return (self.root / Path(path).expanduser()).resolve()


def apply_mode(config: AppConfig, mode: str) -> AppConfig:
    """Retain old cash-mode aliases without replacing the configured checkpoint."""
    if mode in {"money", "trained"}:
        return config
    raise ValueError(f"Retired detection mode: {mode}. Use a trained YOLO26m cash checkpoint.")


def _number(value: object, name: str, minimum: float, maximum: float = math.inf) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{name} must be a number between {minimum} and {maximum}")


def _integer(value: object, name: str, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def validate(config: AppConfig) -> AppConfig:
    d, c, t, a = config.detector, config.camera, config.trigger, config.audio
    if not isinstance(d.model, str) or not d.model.strip():
        raise ValueError("detector.model must be a nonempty path")
    if not isinstance(d.device, str) or not d.device.strip():
        raise ValueError("detector.device must be a nonempty string")
    if d.backend != "trained":
        raise ValueError("detector.backend must be trained; prompt/reference models are retired")
    _number(d.confidence, "detector.confidence", 0.0, 1.0)
    _integer(d.image_size, "detector.image_size", 32)
    if d.image_size % 32:
        raise ValueError("detector.image_size must be a multiple of 32")
    _integer(c.index, "camera.index", 0)
    _integer(c.width, "camera.width", 1)
    _integer(c.height, "camera.height", 1)
    _number(c.max_result_age_seconds, "camera.max_result_age_seconds", 0.01)
    _integer(t.consecutive_hits, "trigger.consecutive_hits", 1)
    _number(t.reset_seconds, "trigger.reset_seconds", 0.01)
    _number(t.cooldown_seconds, "trigger.cooldown_seconds", 0.0)
    _number(a.volume, "audio.volume", 0.0, 1.0)
    if not isinstance(a.file, str) or not a.file.strip():
        raise ValueError("audio.file must be a nonempty path")
    for name, value in [
        ("camera.mirror", c.mirror),
        ("audio.enabled", a.enabled),
        ("overlay.solid_box", config.overlay.solid_box),
    ]:
        if type(value) is not bool:
            raise ValueError(f"{name} must be true or false")
    return config


def load_config(path: Path) -> AppConfig:
    path = path.expanduser().resolve()
    with path.open("rb") as source:
        raw = tomllib.load(source)
    sections = {
        "detector": DetectorConfig,
        "camera": CameraConfig,
        "trigger": TriggerConfig,
        "audio": AudioConfig,
        "overlay": OverlayConfig,
    }
    unknown = set(raw) - set(sections)
    if unknown:
        raise ValueError(f"Unknown config sections: {', '.join(sorted(unknown))}")
    values = {}
    for name, cls in sections.items():
        settings = raw.get(name, {})
        if not isinstance(settings, dict):
            raise ValueError(f"{name} must be a TOML table")
        try:
            values[name] = cls(**settings)
        except TypeError as exc:
            raise ValueError(f"Invalid {name} settings: {exc}") from exc
    return validate(AppConfig(root=path.parent, **values))
