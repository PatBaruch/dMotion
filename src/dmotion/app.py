"""Camera preview and saved-image test entry points."""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from dmotion.audio import AlertSound
from dmotion.config import AppConfig
from dmotion.detector import Detection, MoneyDetector, mirror_detections
from dmotion.monitor import InferenceMonitor
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


def run_camera(config: AppConfig, *, demo: bool = False, checking: bool = False) -> int:
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
        monitor = InferenceMonitor(
            config.camera.max_result_age_seconds, DetectionTrigger(config.trigger)
        )
        future = None
        submitted_at = 0.0
        detections: list[Detection] = []
        demo_until = 0.0
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        logger.info("Q / Esc: quit | M: mute | T: test alert | S: save raw test photo")
        if demo:
            logger.info("DEMO mode: press T for a simulated box; no model runs")
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Camera stopped returning frames. Reconnect it and try again.")
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
                    try:
                        new_detections = future.result()
                    except Exception as exc:
                        monitor.error = str(exc)
                        logger.exception("Detection stopped; try restarting with --device cpu")
                    else:
                        if monitor.complete(new_detections, submitted_at, now):
                            sound.play()
                            logger.info("Object detected" if checking else "Motion detected")
                        if monitor.last_result_stale:
                            logger.warning(
                                "Result discarded: %.0f ms exceeds the %.1fs age limit",
                                monitor.inference_ms,
                                config.camera.max_result_age_seconds,
                            )
                    future = None
                detections = monitor.visible(now) if monitor.error is None else []
                if future is None and monitor.error is None:
                    # One in-flight frame; inference cannot build a queue of old frames.
                    submitted_at = now
                    future = pool.submit(detector.predict, frame.copy())

            display = cv2.flip(frame, 1) if config.camera.mirror else frame.copy()
            display_detections = (
                mirror_detections(detections, frame.shape[1])
                if config.camera.mirror
                else detections
            )
            draw_detections(display, display_detections, solid=config.overlay.solid_box)
            audio_status = "ON" if sound.enabled and sound.sound is not None else "OFF"
            if demo:
                lines = ["DEMO: simulated boxes, no AI | T to simulate"]
            else:
                lines = [
                    "AI CHECK: person / phone / cup / bottle / book"
                    if checking
                    else (
                        "TRAINED MONEY SPREAD | Test with new examples"
                        if config.detector.backend == "trained"
                        else "MONEY MODE | To test common objects: make diagnose"
                    ),
                    monitor.headline(now, checking=checking, confidence=config.detector.confidence),
                    f"Frames analysed: {monitor.completed} | Objects now: {len(detections)}"
                    f" | Last check: {monitor.inference_ms:.0f} ms | Device: {detector.device}",
                ]
                if detector.training_note:
                    lines.append(detector.training_note)
                if monitor.error:
                    lines.extend([monitor.error[:90], "Restart with CPU if this is a device error"])
                elif monitor.last_result_stale:
                    lines.append(
                        "Try CPU or increase camera.max_result_age_seconds"
                        f" | Discarded: {monitor.discarded}"
                    )
            lines.append(f"Audio: {audio_status} | Q quit | M mute | T test | S save photo")
            for index, text in enumerate(lines):
                y = 28 + index * 27
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
