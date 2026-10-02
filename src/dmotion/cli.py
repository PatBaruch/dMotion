"""Command-line entry point; --help and config validation need no ML imports."""

import argparse
import json
import logging
import sys
import time
from dataclasses import replace
from importlib import metadata, util
from pathlib import Path

from dmotion import __version__
from dmotion.config import apply_mode, load_config, validate


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
        ("dataset", "Show photo and label counts"),
        ("label", "Review and draw money-spread boxes in your browser"),
        ("collect", "Save webcam examples automatically"),
        ("import", "Import photos, folders, or video frames"),
        ("fetch", "Download a curated JSON list of image URLs"),
        ("build-dataset", "Export reviewed examples to grouped YOLO splits"),
        ("train", "Train a small money-spread model from reviewed examples"),
    ]:
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", type=Path, default=Path("config.toml"))
        if name in {"run", "image", "prepare"}:
            command.add_argument(
                "--mode",
                choices=("money", "check", "trained"),
                default="money",
                help="check detects common objects to verify the model works",
            )
            command.add_argument("--confidence", type=float)
            command.add_argument(
                "--image-size", type=int, help="Inference image size; use a multiple of 32"
            )
            command.add_argument("--device")
            command.add_argument("--model", help="Override the model path for the selected mode")
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
        if name in {"dataset", "label", "collect", "import", "fetch", "build-dataset", "train"}:
            command.add_argument("--dataset", type=Path, default=Path("data/training"))
        if name == "label":
            command.add_argument("--port", type=int, default=0)
            command.add_argument("--no-browser", action="store_true")
        if name == "collect":
            command.add_argument("--seconds", type=float, default=20)
            command.add_argument("--interval", type=float, default=1)
            command.add_argument("--camera", type=int)
            command.add_argument(
                "--kind", choices=("unreviewed", "positive", "negative"), default="unreviewed"
            )
        if name == "import":
            command.add_argument("paths", type=Path, nargs="+")
            command.add_argument("--group")
            command.add_argument("--interval", type=float, default=1)
        if name == "fetch":
            command.add_argument("sources", type=Path)
            command.add_argument("--limit", type=int, default=20)
        if name == "train":
            command.add_argument("--epochs", type=int, default=30)
            command.add_argument("--image-size", type=int, default=640)
            command.add_argument("--device", default="auto")
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
        if getattr(args, "mode", None):
            config = apply_mode(config, args.mode)
            if args.mode == "check":
                logging.info("AI check mode: look for a person, phone, cup, bottle, or book")
        overrides = {}
        for name in ("confidence", "image_size", "device", "model"):
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
        if hasattr(args, "dataset"):
            from dmotion.dataset import Dataset, export_dataset

            directory = config.resolve(str(args.dataset))
            if args.command == "dataset":
                print(json.dumps(Dataset(directory).summary(), indent=2))
                return 0
            if args.command == "label":
                from dmotion.labeler import run_labeler

                return run_labeler(directory, open_browser=not args.no_browser, port=args.port)
            if args.command == "collect":
                from dmotion.collect import record_camera

                record_camera(
                    config, directory, seconds=args.seconds, interval=args.interval, kind=args.kind
                )
                return 0
            if args.command == "import":
                from dmotion.collect import import_images, import_video

                count = 0
                photos = []
                for path in args.paths:
                    path = path.expanduser().resolve()
                    if path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}:
                        if args.group:
                            raise ValueError(
                                "Video groups are automatic; omit --group for video import"
                            )
                        count += import_video(directory, path, interval=args.interval)
                    else:
                        photos.append(path)
                if photos:
                    count += import_images(directory, photos, group=args.group)
                print(f"Imported {count} new examples. Run make label.")
                return 0
            if args.command == "fetch":
                from dmotion.download import fetch_images

                report = fetch_images(directory, args.sources, limit=args.limit)
                return 0 if report["downloaded"] else 1
            if args.command == "build-dataset":
                print(export_dataset(Dataset(directory), config.root / "data/yolo"))
                return 0
            if args.command == "train":
                from dmotion.training import train_model

                model = train_model(
                    config,
                    directory,
                    epochs=args.epochs,
                    image_size=args.image_size,
                    device=args.device,
                )
                print(f"Saved {model}. Test it with make trained.")
                return 0
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
        return run_camera(config, demo=args.demo, checking=args.mode == "check")
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        logging.error("%s", exc)
        if isinstance(exc, ImportError):
            logging.error("Install the vision dependencies with: make setup")
        else:
            logging.info("For setup diagnostics, run: make doctor")
        return 1
