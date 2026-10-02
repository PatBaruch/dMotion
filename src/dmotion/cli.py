"""Command-line entry point; --help and config validation need no ML imports."""

import argparse
import logging
import sys
import time
from dataclasses import replace
from importlib import metadata, util
from pathlib import Path

from dmotion import __version__
from dmotion.config import load_config, validate


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Local money-spread camera experiment")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    for name, help_text in [
        ("run", "Open the webcam detector"),
        ("image", "Detect a saved photo and save an annotated copy"),
        ("prepare", "Download/load model weights and verify inference without a camera"),
        ("doctor", "Validate settings and report installed dependencies"),
        ("sound", "Test the configured alert sound"),
    ]:
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", type=Path, default=Path("config.toml"))
        if name in {"run", "image", "prepare"}:
            command.add_argument("--confidence", type=float)
            command.add_argument("--device")
            command.add_argument(
                "--prompt", action="append", help="Repeat to replace default prompts"
            )
        if name == "run":
            command.add_argument("--camera", type=int)
            command.add_argument("--mute", action="store_true")
            command.add_argument("--demo", action="store_true", help="Manual box/sound test; no AI")
        if name == "image":
            command.add_argument("source", type=Path)
            command.add_argument("--output", type=Path)
            command.add_argument("--show", action="store_true")
    return root


def doctor(config) -> int:
    print(f"dMotion {__version__} | Python {sys.version.split()[0]}")
    print(f"Project: {config.root}\nConfiguration: valid")
    missing = []
    for module, distribution in [
        ("cv2", "opencv-python"),
        ("ultralytics", "ultralytics"),
        ("torch", "torch"),
        ("pygame", "pygame"),
        ("clip", "clip"),
    ]:
        if util.find_spec(module) is None:
            print(f"{distribution}: MISSING")
            missing.append(distribution)
        else:
            print(f"{distribution}: {metadata.version(distribution)}")
    print(f"Model: {config.resolve(config.detector.model)}")
    print(f"Prompts: {', '.join(config.detector.prompts)}")
    sound = config.resolve(config.audio.file)
    print(f"Sound: {sound if sound.is_file() else 'built-in beep (custom WAV not added)'}")
    if missing:
        print("Install dependencies with: make setup", file=sys.stderr)
    return 1 if missing else 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        config = load_config(args.config)
        overrides = {}
        for name in ("confidence", "device"):
            if getattr(args, name, None) is not None:
                overrides[name] = getattr(args, name)
        if getattr(args, "prompt", None):
            overrides["prompts"] = tuple(args.prompt)
        if overrides:
            config = replace(config, detector=replace(config.detector, **overrides))
        if getattr(args, "camera", None) is not None:
            config = replace(config, camera=replace(config.camera, index=args.camera))
        if getattr(args, "mute", False):
            config = replace(config, audio=replace(config.audio, enabled=False))
        validate(config)
        if args.command == "doctor":
            return doctor(config)
        if args.command == "prepare":
            import numpy as np

            from dmotion.detector import MoneyDetector

            detector = MoneyDetector(config)
            start = time.perf_counter()
            detector.predict(np.zeros((480, 640, 3), dtype=np.uint8))
            print(
                f"Model ready on {detector.device}; warm-up took {time.perf_counter() - start:.2f}s"
            )
            return 0
        if args.command == "sound":
            from dmotion.audio import AlertSound

            sound = AlertSound(config)
            try:
                if not sound.play(force=True):
                    raise RuntimeError("No audio device available")
                time.sleep(sound.sound.get_length() + 0.1)
            finally:
                sound.close()
            return 0
        from dmotion.app import run_camera, run_image

        if args.command == "image":
            return run_image(config, args.source, args.output, args.show)
        return run_camera(config, demo=args.demo)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        logging.error("%s", exc)
        if isinstance(exc, ImportError):
            logging.error("Install the vision dependencies with: make setup")
        else:
            logging.info("For setup diagnostics, run: make doctor")
        return 1
