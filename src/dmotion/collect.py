"""Collect real examples; all new frames require review before training."""

import hashlib
import math
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

from .config import AppConfig
from .dataset import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _positive_number(value: float, name: str) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive number")


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_images(directory: Path, paths: list[Path], *, group: str | None = None) -> int:
    """Import files/folders. Use a shared group for frames from the same recording."""
    if not paths:
        raise ValueError("Provide at least one image file or folder")
    files = set()
    for supplied in paths:
        path = supplied.expanduser().resolve()
        if path.is_dir():
            files.update(
                item
                for item in path.rglob("*")
                if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
            )
        elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            files.add(path)
        else:
            raise ValueError(f"Not an image file or folder: {path}")
    if not files:
        raise ValueError("No supported images found in the supplied folders")
    dataset = Dataset(directory)
    before = len(dataset.records())
    for path in sorted(files):
        image_group = group if group is not None else f"photo-{_file_hash(path)[:16]}"
        dataset.add_image(path, group=image_group, source=f"Local image: {path.name}")
    added = len(dataset.records()) - before
    print(f"Imported {added} new images. Review them with the label command.")
    return added


def record_camera(
    config: AppConfig,
    directory: Path,
    *,
    seconds: float = 20,
    interval: float = 1,
    camera: int | None = None,
    kind: str = "unreviewed",
) -> int:
    """Sample unmirrored frames from one camera session, with a live preview."""
    _positive_number(seconds, "seconds")
    _positive_number(interval, "interval")
    if kind not in {"unreviewed", "positive", "negative"}:
        raise ValueError("kind must be unreviewed, positive, or negative")
    if camera is not None and (type(camera) is not int or camera < 0):
        raise ValueError("camera must be an integer >= 0")
    import cv2

    dataset = Dataset(directory)
    directory.mkdir(parents=True, exist_ok=True)
    session = f"camera-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
    before = len(dataset.records())
    capture = cv2.VideoCapture(config.camera.index if camera is None else camera)
    window = "dMotion - collect training examples"
    try:
        if not capture.isOpened():
            raise RuntimeError("Cannot open camera. Check the camera index and camera permission.")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.height)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        started = time.monotonic()
        next_sample = started
        saved = 0
        with tempfile.TemporaryDirectory(prefix="capture-", dir=directory) as temporary:
            while time.monotonic() - started < seconds:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("The camera stopped returning frames")
                now = time.monotonic()
                if now >= next_sample:
                    path = Path(temporary) / f"frame-{saved:05}.jpg"
                    if not cv2.imwrite(str(path), frame):
                        raise RuntimeError("Could not save a camera frame")
                    height, width = frame.shape[:2]
                    dataset.add_image(
                        path,
                        group=session,
                        source=f"Camera recording ({kind} hint; needs review)",
                        width=width,
                        height=height,
                    )
                    saved += 1
                    next_sample = now + interval
                preview = cv2.flip(frame, 1) if config.camera.mirror else frame.copy()
                remaining = max(0, seconds - (now - started))
                lines = [
                    f"Recording: {remaining:.0f}s left | {saved} frames saved",
                    (
                        "Show empty hands, single bills, stacks and cards. Q = finish."
                        if kind == "negative"
                        else "Move the fan; change angle, distance and lighting. Q = finish."
                    ),
                    f"{kind.capitalize()} examples | Review every saved frame before training.",
                ]
                for index, line in enumerate(lines):
                    position = (14, 28 + index * 28)
                    cv2.putText(
                        preview, line, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4
                    )
                    cv2.putText(
                        preview, line, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
                    )
                cv2.imshow(window, preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        capture.release()
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass
    added = len(dataset.records()) - before
    print(f"Saved {added} new images in session {session}. Review them with the label command.")
    return added


def import_video(directory: Path, video: Path, *, interval: float = 1) -> int:
    """Sample a video, keeping every frame in one group to prevent split leakage."""
    _positive_number(interval, "interval")
    video = video.expanduser().resolve()
    if not video.is_file():
        raise ValueError(f"Video file does not exist: {video}")
    import cv2

    dataset = Dataset(directory)
    directory.mkdir(parents=True, exist_ok=True)
    before = len(dataset.records())
    group = f"video-{_file_hash(video)[:16]}"
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video has no usable frame rate; convert it to a regular MP4 first")
        step = max(1, round(fps * interval))
        frame_index = 0
        with tempfile.TemporaryDirectory(prefix="video-", dir=directory) as temporary:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % step == 0:
                    path = Path(temporary) / f"frame-{frame_index:08}.jpg"
                    if not cv2.imwrite(str(path), frame):
                        raise RuntimeError("Could not save a video frame")
                    height, width = frame.shape[:2]
                    dataset.add_image(
                        path,
                        group=group,
                        source=f"Video: {video.name}, {frame_index / fps:.2f}s",
                        width=width,
                        height=height,
                    )
                frame_index += 1
    finally:
        capture.release()
    added = len(dataset.records()) - before
    if frame_index == 0:
        raise ValueError("Video contains no readable frames")
    print(f"Imported {added} new frames. Review them with the label command.")
    return added
