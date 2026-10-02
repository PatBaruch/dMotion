"""A recording's related frames must remain unreviewed and in one source group."""

import sys
from pathlib import Path
from types import SimpleNamespace

from dmotion.collect import import_video, record_camera
from dmotion.config import AppConfig, CameraConfig
from dmotion.dataset import Dataset


def test_video_frames_keep_one_group_and_need_review(tmp_path, monkeypatch):
    video = tmp_path / "test-recording.mp4"
    video.write_bytes(b"one original video")
    frames = [SimpleNamespace(shape=(60, 100, 3), index=index) for index in range(3)]

    class Capture:
        def __init__(self, _path):
            self.frames = iter(frames)

        def isOpened(self):
            return True

        def get(self, _property):
            return 1.0

        def read(self):
            frame = next(self.frames, None)
            return frame is not None, frame

        def release(self):
            pass

    def write(path, frame):
        Path(path).write_bytes(f"encoded frame {frame.index}".encode())
        return True

    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(VideoCapture=Capture, CAP_PROP_FPS=5, imwrite=write),
    )
    directory = tmp_path / "dataset"
    assert import_video(directory, video, interval=1) == 3
    records = Dataset(directory).records()
    assert len({record["group"] for record in records}) == 1
    assert all(record["status"] == "unreviewed" and record["boxes"] == [] for record in records)
    assert all(record["source"].startswith("Video: test-recording.mp4,") for record in records)
    assert import_video(directory, video, interval=1) == 0


def test_camera_saves_raw_frames_and_unreviewed_hints_then_cleans_up(tmp_path, monkeypatch):
    frames = [
        SimpleNamespace(shape=(60, 100, 3), index=index, mirrored=False, overlays=[])
        for index in range(2)
    ]
    previews = []
    closed_windows = []
    keys = iter([-1, ord("q")])
    ticks = iter([0.0, 0.1, 0.1, 1.2, 1.2])
    monkeypatch.setattr("dmotion.collect.time", SimpleNamespace(monotonic=lambda: next(ticks)))

    class Capture:
        released = False

        def __init__(self):
            self.frames = iter(frames)

        def isOpened(self):
            return True

        def set(self, _property, _value):
            pass

        def read(self):
            return True, next(self.frames)

        def release(self):
            self.released = True

    capture = Capture()

    def write(path, frame):
        assert frame in frames and not frame.mirrored and not frame.overlays
        Path(path).write_bytes(f"raw camera frame {frame.index}".encode())
        return True

    def flip(frame, _direction):
        return SimpleNamespace(shape=frame.shape, index=frame.index, mirrored=True, overlays=[])

    def overlay(frame, text, *_args):
        frame.overlays.append(text)

    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(
            VideoCapture=lambda _index: capture,
            CAP_PROP_FRAME_WIDTH=3,
            CAP_PROP_FRAME_HEIGHT=4,
            WINDOW_NORMAL=0,
            FONT_HERSHEY_SIMPLEX=0,
            WND_PROP_VISIBLE=0,
            namedWindow=lambda *_args: None,
            imwrite=write,
            flip=flip,
            putText=overlay,
            imshow=lambda _window, frame: previews.append(frame),
            waitKey=lambda _delay: next(keys),
            getWindowProperty=lambda *_args: 1,
            destroyWindow=closed_windows.append,
            error=RuntimeError,
        ),
    )
    directory = tmp_path / "dataset"
    config = AppConfig(root=tmp_path, camera=CameraConfig(mirror=True))
    assert record_camera(config, directory, seconds=20, interval=1, kind="negative") == 2
    dataset = Dataset(directory)
    records = dataset.records()
    assert len({record["group"] for record in records}) == 1
    assert all(record["status"] == "unreviewed" and record["boxes"] == [] for record in records)
    assert all("negative hint; needs review" in record["source"] for record in records)
    assert {dataset.image_path(record).read_bytes() for record in records} == {
        b"raw camera frame 0",
        b"raw camera frame 1",
    }
    assert len(previews) == 2 and all(frame.mirrored and frame.overlays for frame in previews)
    assert all(not frame.overlays for frame in frames)
    assert capture.released and closed_windows == ["dMotion - collect training examples"]
    assert not list(directory.glob("capture-*"))
