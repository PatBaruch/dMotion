"""Camera preview and saved-image test entry points."""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from dmotion.audio import AlertSound
from dmotion.config import AppConfig
from dmotion.detector import Detection, MoneyDetector
from dmotion.trigger import DetectionTrigger

logger = logging.getLogger(__name__)
WINDOW = "dMotion"


def draw_detections(frame, detections: list[Detection], *, solid: bool = False):
    import cv2

    for detection in detections:
        x1, y1, x2, y2 = detection.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1 if solid else 2)
        label = f"{detection.label}: {detection.confidence:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return frame


def run_image(config: AppConfig, source: Path, output: Path | None, show: bool) -> int:
    import cv2

    frame = cv2.imread(str(source))
    if frame is None:
        raise ValueError(f"Cannot read image: {source}")
    detector = MoneyDetector(config)
    start = time.perf_counter()
    detections = detector.predict(frame)
    elapsed = time.perf_counter() - start
    target = output or config.root / "outputs" / f"{source.stem}-detected.jpg"
    if target.resolve() == source.resolve():
        raise ValueError("Choose an output path different from the source image")
    target.parent.mkdir(parents=True, exist_ok=True)
    draw_detections(frame, detections, solid=config.overlay.solid_box)
    if not cv2.imwrite(str(target), frame):
        raise OSError(f"Could not write image: {target}")
    print(
        json.dumps(
            {
                "detections": [
                    {"label": d.label, "confidence": round(d.confidence, 4), "box": d.box}
                    for d in detections
                ],
                "inference_seconds": round(elapsed, 3),
                "output": str(target.resolve()),
            },
            indent=2,
        )
    )
    if show:
        try:
            cv2.imshow(WINDOW, frame)
            cv2.waitKey(0)
        finally:
            cv2.destroyAllWindows()
    return 0


def run_camera(config: AppConfig, *, demo: bool = False) -> int:
    import cv2

    detector = None if demo else MoneyDetector(config)
    capture = cv2.VideoCapture(config.camera.index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            "Cannot open camera. Allow Camera access for your terminal/Codex in macOS "
            "System Settings > Privacy & Security > Camera, close other camera apps, "
            "or try --camera 1."
        )
    sound = None
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.height)
        sound = AlertSound(config)
        trigger = DetectionTrigger(config.trigger)
        future = None
        submitted_at = 0.0
        result_at = float("-inf")
        detections: list[Detection] = []
        inference_ms = 0.0
        demo_until = 0.0
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        logger.info("Q / Esc: quit | M: mute | T: test alert | S: save raw test photo")
        if demo:
            logger.info("DEMO mode: press T for a simulated box; no model runs")
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Camera stopped returning frames. Reconnect it and try again.")
            if config.camera.mirror:
                frame = cv2.flip(frame, 1)
            now = time.monotonic()
            if demo:
                height, width = frame.shape[:2]
                detections = (
                    [
                        Detection(
                            (width // 3, height // 3, width * 2 // 3, height * 2 // 3),
                            1.0,
                            "SIMULATED spread",
                        )
                    ]
                    if now < demo_until
                    else []
                )
            else:
                if future is not None and future.done():
                    new_detections = future.result()
                    inference_ms = (now - submitted_at) * 1000
                    fresh = now - submitted_at <= config.camera.max_result_age_seconds
                    detections = new_detections if fresh else []
                    result_at = submitted_at
                    if trigger.update(bool(detections), now):
                        sound.play()
                        logger.info("Motion detected")
                    future = None
                if now - result_at > config.camera.max_result_age_seconds:
                    detections = []
                    trigger.update(False, now)
                if future is None:
                    # One in-flight frame; inference cannot build a queue of old frames.
                    submitted_at = now
                    future = pool.submit(detector.predict, frame.copy())

            display = draw_detections(frame.copy(), detections, solid=config.overlay.solid_box)
            mode = (
                "DEMO / T to simulate"
                if demo
                else ("MONEY DETECTED" if detections else "Looking for money")
            )
            audio_status = "ON" if sound.enabled and sound.sound is not None else "OFF"
            for y, text in [
                (28, mode),
                (55, f"Audio: {audio_status} | {inference_ms:.0f} ms"),
                (82, "Q quit | M mute | T test | S save photo"),
            ]:
                cv2.putText(
                    display, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA
                )
                cv2.putText(
                    display,
                    text,
                    (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            cv2.imshow(WINDOW, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("m"):
                sound.enabled = not sound.enabled
            elif key == ord("t"):
                sound.play(force=True)
                if demo:
                    demo_until = now + 2.0
            elif key == ord("s"):
                photos = config.root / "data"
                photos.mkdir(parents=True, exist_ok=True)
                target = photos / f"test-{datetime.now():%Y%m%d-%H%M%S-%f}.jpg"
                if not cv2.imwrite(str(target), frame):
                    raise OSError(f"Could not save test photo: {target}")
                logger.info("Saved %s", target)
    finally:
        capture.release()
        cv2.destroyAllWindows()
        pool.shutdown(wait=True, cancel_futures=True)
        if sound is not None:
            sound.close()
    return 0
