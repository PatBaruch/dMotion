"""Reviewed image records and reproducible, session-separated YOLO exports.

Only explicitly reviewed positives and negatives are exported. A missing box must
never silently turn an unreviewed internet photo into a negative example.
"""

import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from collections import Counter
from copy import deepcopy
from functools import wraps
from io import BytesIO
from pathlib import Path

from filelock import FileLock

STATUSES = {"unreviewed", "positive", "negative", "excluded"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _serialized_mutation(method):
    """Serialize complete manifest updates across browser, import and AI processes."""

    @wraps(method)
    def locked(self, *args, **kwargs):
        with self._manifest_lock:
            return method(self, *args, **kwargs)

    return locked


def _validate_suggestion(suggestion: dict, width: int, height: int) -> None:
    if not isinstance(suggestion, dict):
        raise ValueError("An AI suggestion must be an object")
    boxes = suggestion.get("boxes")
    _validate_boxes("positive" if boxes else "unreviewed", boxes, width, height)
    scores, labels = suggestion.get("scores"), suggestion.get("labels")
    if not isinstance(scores, list) or len(scores) != len(boxes):
        raise ValueError("Each suggested box needs a confidence score")
    if any(
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not 0 <= score <= 1
        or not math.isfinite(score)
        for score in scores
    ):
        raise ValueError("Suggestion confidence scores must be between 0 and 1")
    if not isinstance(labels, list) or len(labels) != len(boxes):
        raise ValueError("Each suggested box needs a label")
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("Suggestion labels must be nonempty strings")
    for field in ("model", "created_at"):
        if not isinstance(suggestion.get(field), str) or not suggestion[field].strip():
            raise ValueError(f"Suggestion {field} must be a nonempty string")
    prompts = suggestion.get("prompts")
    if (
        not isinstance(prompts, list)
        or not prompts
        or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts)
    ):
        raise ValueError("Suggestion prompts must be nonempty strings")


def _write_json(path: Path, value: object) -> None:
    """Replace a JSON file only after its complete contents have been written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _validate_boxes(status: str, boxes: list[list[int]], width: int, height: int) -> None:
    if status not in STATUSES:
        raise ValueError(f"Unknown review status: {status}")
    if not isinstance(boxes, list):
        raise ValueError("Boxes must be a list of pixel rectangles")
    if status != "positive" and boxes:
        raise ValueError(f"{status} images cannot have boxes")
    if status == "positive" and not boxes:
        raise ValueError("A positive image needs a box around each cash bundle or single bill")
    for box in boxes:
        if not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box):
            raise ValueError("Each box must contain four integer pixel coordinates: x1,y1,x2,y2")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"Box {box} must have positive area and fit inside {width}x{height}")


class Dataset:
    def __init__(self, directory: Path):
        self.directory = Path(directory).expanduser().resolve()
        self.manifest_path = self.directory / "manifest.json"
        (self.directory / "images").mkdir(parents=True, exist_ok=True)
        self._manifest_lock = FileLock(self.directory / ".manifest.lock")
        with self._manifest_lock:
            if not self.manifest_path.exists():
                _write_json(self.manifest_path, {"version": 1, "records": []})
            self.records()  # Report a corrupt or incompatible manifest immediately.

    def records(self) -> list[dict]:
        raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("version") != 1:
            raise ValueError("Unsupported dataset manifest; expected version 1")
        records = raw.get("records")
        if not isinstance(records, list):
            raise ValueError("Dataset manifest must contain an image records list")
        ids: set[str] = set()
        hashes: set[str] = set()
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("Each dataset record must be an object")
            for field in ("id", "image", "group", "source", "sha256", "status"):
                if not isinstance(record.get(field), str):
                    raise ValueError(f"Dataset record {field} must be a string")
            if not record["id"] or not record["group"].strip():
                raise ValueError("Every image needs an ID and a nonempty session/source group")
            if record["id"] in ids or record["sha256"] in hashes:
                raise ValueError("Dataset manifest contains duplicate image IDs or hashes")
            ids.add(record["id"])
            hashes.add(record["sha256"])
            for field in ("width", "height"):
                if type(record.get(field)) is not int or record[field] < 1:
                    raise ValueError(f"Dataset record {field} must be a positive integer")
            self.image_path(record)
            _validate_boxes(
                record["status"], record.get("boxes"), record["width"], record["height"]
            )
            if "suggestion" in record:
                _validate_suggestion(record["suggestion"], record["width"], record["height"])
        return records

    def get_record(self, id: str) -> dict:
        for record in self.records():
            if record["id"] == id:
                return record
        raise ValueError(f"No dataset image with ID {id}")

    def image_path(self, record: dict) -> Path:
        relative = Path(record["image"])
        target = (self.directory / relative).resolve()
        if relative.is_absolute() or not target.is_relative_to(self.directory / "images"):
            raise ValueError("Dataset image paths must stay inside the images folder")
        return target

    @_serialized_mutation
    def add_image(
        self,
        path: Path,
        *,
        group: str,
        source: str = "",
        width: int | None = None,
        height: int | None = None,
    ) -> dict:
        path = Path(path).expanduser().resolve()
        if not isinstance(group, str) or not group.strip():
            raise ValueError("Give every recording/source a nonempty group name")
        if not isinstance(source, str):
            raise ValueError("Image source must be a string")
        suffix = path.suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image type: {suffix or '(no extension)'}")
        contents = path.read_bytes()
        digest = hashlib.sha256(contents).hexdigest()
        records = self.records()
        for record in records:
            if record["sha256"] == digest:
                return record
        if width is None or height is None:
            from PIL import Image, ImageOps

            with Image.open(path) as image:
                # Verify decoding rather than importing a broken image into the labeling queue.
                image.load()
                width, height = image.size
                if image.getexif().get(274, 1) in range(2, 9):
                    normalized = ImageOps.exif_transpose(image)
                    width, height = normalized.size
                    buffer = BytesIO()
                    normalized.save(buffer, format=image.format)
                    contents = buffer.getvalue()
                    digest = hashlib.sha256(contents).hexdigest()
                    for record in records:
                        if record["sha256"] == digest:
                            return record
        if any(type(value) is not int or value < 1 for value in (width, height)):
            raise ValueError("Image dimensions must be positive integers")
        id = digest[:16]
        if any(record["id"] == id for record in records):
            id = digest
        relative = Path("images") / f"{id}{suffix}"
        destination = self.directory / relative
        if destination != path or contents != path.read_bytes():
            destination.write_bytes(contents)
        record = {
            "id": id,
            "image": relative.as_posix(),
            "width": width,
            "height": height,
            "group": group.strip(),
            "source": source,
            "sha256": digest,
            "status": "unreviewed",
            "boxes": [],
        }
        records.append(record)
        _write_json(self.manifest_path, {"version": 1, "records": records})
        return deepcopy(record)

    @_serialized_mutation
    def review(self, id: str, *, status: str, boxes: list[list[int]]) -> dict:
        records = self.records()
        for record in records:
            if record["id"] != id:
                continue
            _validate_boxes(status, boxes, record["width"], record["height"])
            record["status"] = status
            record["boxes"] = deepcopy(boxes)
            _write_json(self.manifest_path, {"version": 1, "records": records})
            return deepcopy(record)
        raise ValueError(f"No dataset image with ID {id}")

    @_serialized_mutation
    def suggest(self, id: str, suggestion: dict) -> dict:
        """Save AI drafts without changing reviewed labels or making training examples."""
        records = self.records()
        for record in records:
            if record["id"] != id:
                continue
            if record["status"] != "unreviewed":
                raise ValueError("AI suggestions cannot replace a reviewed picture")
            _validate_suggestion(suggestion, record["width"], record["height"])
            record["suggestion"] = deepcopy(suggestion)
            _write_json(self.manifest_path, {"version": 1, "records": records})
            return deepcopy(record)
        raise ValueError(f"No dataset image with ID {id}")

    def summary(self) -> dict:
        records = self.records()
        counts = Counter(record["status"] for record in records)
        return {
            "total": len(records),
            **{status: counts[status] for status in sorted(STATUSES)},
            "reviewed": counts["positive"] + counts["negative"],
            "groups": len({record["group"] for record in records}),
            "positive_groups": len(
                {record["group"] for record in records if record["status"] == "positive"}
            ),
            "ai_suggested": sum(
                record["status"] == "unreviewed" and "suggestion" in record for record in records
            ),
        }


def _split_groups(records: list[dict], seed: int) -> dict[str, str]:
    positive = sorted({record["group"] for record in records if record["status"] == "positive"})
    if len(positive) < 3:
        raise ValueError(
            "Training needs reviewed cash from at least 3 independent groups "
            "(different recording sessions or source shoots). Review more images first."
        )
    randomizer = random.Random(seed)
    randomizer.shuffle(positive)
    train_count = max(1, min(len(positive) - 2, int(len(positive) * 0.70)))
    val_count = max(1, min(len(positive) - train_count - 1, int(len(positive) * 0.15)))
    mapping = {
        group: "train"
        if index < train_count
        else "val"
        if index < train_count + val_count
        else "test"
        for index, group in enumerate(positive)
    }
    negative_only = sorted({record["group"] for record in records} - set(positive))
    randomizer.shuffle(negative_only)
    # Negative-only sessions also stay intact. When possible, put one in each split.
    if len(negative_only) >= 3:
        count_train = max(1, min(len(negative_only) - 2, int(len(negative_only) * 0.70)))
        count_val = max(
            1, min(len(negative_only) - count_train - 1, int(len(negative_only) * 0.15))
        )
    else:
        count_train, count_val = len(negative_only), 0
    for index, group in enumerate(negative_only):
        mapping[group] = (
            "train" if index < count_train else "val" if index < count_train + count_val else "test"
        )
    return mapping


def export_dataset(dataset: Dataset, output: Path, seed: int = 42) -> Path:
    """Build labels only from reviewed records, splitting by complete sessions."""
    records = [r for r in dataset.records() if r["status"] in {"positive", "negative"}]
    mapping = _split_groups(records, seed)
    output = Path(output).expanduser().resolve()
    if dataset.directory.is_relative_to(output) or output.is_relative_to(
        dataset.directory / "images"
    ):
        raise ValueError("Export folder must not replace the source dataset or its images")
    report_path = output / "export-report.json"
    if output.exists() and any(output.iterdir()) and not report_path.is_file():
        raise ValueError(f"Refusing to replace a non-export folder: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    backup = staging.with_name(staging.name + "-previous")
    counts = {
        name: {"images": 0, "positive": 0, "negative": 0, "boxes": 0, "groups": 0}
        for name in ("train", "val", "test")
    }
    provenance = []
    try:
        for name in counts:
            (staging / "images" / name).mkdir(parents=True)
            (staging / "labels" / name).mkdir(parents=True)
            counts[name]["groups"] = sum(split == name for split in mapping.values())
        for record in records:
            source = dataset.image_path(record)
            if hashlib.sha256(source.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"Image changed since import: {source.name}; import it again")
            split = mapping[record["group"]]
            filename = Path(record["image"]).name
            shutil.copyfile(source, staging / "images" / split / filename)
            lines = []
            for x1, y1, x2, y2 in record["boxes"]:
                values = (
                    (x1 + x2) / (2 * record["width"]),
                    (y1 + y2) / (2 * record["height"]),
                    (x2 - x1) / record["width"],
                    (y2 - y1) / record["height"],
                )
                lines.append("0 " + " ".join(f"{value:.8f}" for value in values))
            (staging / "labels" / split / f"{record['id']}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            counts[split]["images"] += 1
            counts[split][record["status"]] += 1
            counts[split]["boxes"] += len(record["boxes"])
            provenance.append({**record, "split": split})
        warnings = [
            f"{split} has no negative examples; false alarms are not measured well."
            for split, count in counts.items()
            if not count["negative"]
        ]
        if len(records) < 100:
            warnings.append(
                "This is a small starter dataset; test on new webcam sessions before relying on it."
            )
        _write_json(
            staging / "export-report.json",
            {
                "version": 1,
                "seed": seed,
                "source": str(dataset.directory),
                "class": "money_spread",
                "summary": dataset.summary(),
                "splits": counts,
                "warnings": warnings,
                "records": provenance,
            },
        )
        (staging / "dataset.yaml").write_text(
            f"path: {json.dumps(str(output))}\n"
            "train: images/train\nval: images/val\ntest: images/test\n"
            "names:\n  0: money_spread\n",
            encoding="utf-8",
        )
        if output.exists():
            os.replace(output, backup)
        try:
            os.replace(staging, output)
        except BaseException:
            if backup.exists():
                os.replace(backup, output)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output / "dataset.yaml"
