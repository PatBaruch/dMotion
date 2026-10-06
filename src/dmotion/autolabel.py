"""Generate cash-box drafts locally, leaving every result pending review."""

import hashlib
import logging
import math
import re
from datetime import UTC, datetime
from pathlib import Path

from dmotion.config import AppConfig
from dmotion.dataset import Dataset, _write_json
from dmotion.detector import MoneyDetector

logger = logging.getLogger(__name__)


def _overlap(first: list[int], second: list[int]) -> float:
    intersection = max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0, min(first[3], second[3]) - max(first[1], second[1])
    )
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (first_area + second_area - intersection)


def _draft_boxes(detections, width: int, height: int) -> tuple[list, list, list]:
    boxes, scores, labels = [], [], []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if not 0 <= detection.confidence <= 1 or not math.isfinite(detection.confidence):
            raise ValueError("Detector returned an invalid confidence score")
        if len(detection.box) != 4 or any(not math.isfinite(v) for v in detection.box):
            raise ValueError("Detector returned invalid box coordinates")
        x1, y1, x2, y2 = (round(value) for value in detection.box)
        box = [max(0, x1), max(0, y1), min(width, x2), min(height, y2)]
        if box[0] >= box[2] or box[1] >= box[3]:
            continue
        # Cash detections describe the same class. Keep one proposal per overlapping object.
        if any(_overlap(box, kept) >= 0.5 for kept in boxes):
            continue
        boxes.append(box)
        scores.append(float(detection.confidence))
        labels.append(detection.label)
    return boxes, scores, labels


def _contact_sheets(dataset: Dataset, records: list[dict], output: Path) -> list[str]:
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    paths = []
    font = ImageFont.load_default(size=15)
    for page, offset in enumerate(range(0, len(records), 12), start=1):
        subset = records[offset : offset + 12]
        rows = math.ceil(len(subset) / 4)
        sheet = Image.new("RGB", (1200, rows * 420), "#161b24")
        text = ImageDraw.Draw(sheet)
        for index, record in enumerate(subset):
            column, row = index % 4, index // 4
            left, top = column * 300, row * 420
            suggestion = record["suggestion"]
            with Image.open(dataset.image_path(record)) as source:
                annotated = source.convert("RGB")
            drawing = ImageDraw.Draw(annotated)
            for box, score in zip(suggestion["boxes"], suggestion["scores"], strict=True):
                drawing.rectangle(box, outline="#ffcf55", width=5)
                drawing.text((box[0] + 4, box[1] + 4), f"{score:.2f}", fill="#ffcf55")
            thumbnail = ImageOps.contain(annotated, (284, 345))
            sheet.paste(thumbnail, (left + (300 - thumbnail.width) // 2, top + 60))
            source_label = record["source"].removeprefix("Video: ")
            text.text((left + 8, top + 6), source_label, fill="white", font=font)
            count = len(suggestion["boxes"])
            note = (
                f"{count} AI box(es) — needs review"
                if count
                else "No AI box — check for missed cash"
            )
            text.text((left + 8, top + 30), note, fill="#ffcf55", font=font)
        path = output / f"contact-sheet-{page:02}.jpg"
        sheet.save(path, quality=90)
        paths.append(str(path))
    return paths


def auto_label(config: AppConfig, directory: Path, *, output: Path | None = None) -> Path:
    """Persist model suggestions without accepting any positives or negatives."""
    dataset = Dataset(directory)
    unreviewed = [record for record in dataset.records() if record["status"] == "unreviewed"]
    if not unreviewed:
        raise ValueError("No unreviewed pictures. Import your videos before auto-labeling.")
    records = [record for record in unreviewed if "suggestion" in record]
    pending = [record for record in unreviewed if "suggestion" not in record]
    preserved_frames = len(records)
    output = config.resolve("outputs/video-autolabel") if output is None else Path(output).resolve()
    if output.is_relative_to(dataset.directory) or dataset.directory.is_relative_to(output):
        raise ValueError("Auto-label output must be separate from the source dataset")
    output.mkdir(parents=True, exist_ok=True)
    created = datetime.now(UTC).isoformat()
    for record in records:
        source = dataset.image_path(record)
        if hashlib.sha256(source.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Image changed since import: {source.name}")
    # Restore the audit artifact even if the previous run stopped after its last save.
    _write_json(output / "predictions.json", {"version": 1, "records": records})
    model = config.resolve(config.detector.model)
    model_digest = revision = device = None
    if pending:
        import cv2

        if not model.is_file():
            raise ValueError(
                "Supply a trained YOLO26m cash checkpoint with --model before auto-labeling"
            )
        with model.open("rb") as checkpoint:
            model_digest = hashlib.file_digest(checkpoint, "sha256").hexdigest()
        detector = MoneyDetector(config)
        device = detector.device
        revision = f"sha256:{model_digest}"
        logger.info("Suggesting cash boxes for %s pictures on %s", len(pending), device)
    for index, record in enumerate(pending, start=1):
        source = dataset.image_path(record)
        if hashlib.sha256(source.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Image changed since import: {source.name}")
        frame = cv2.imread(str(source))
        if frame is None:
            raise ValueError(f"Cannot decode imported picture: {source.name}")
        height, width = frame.shape[:2]
        if (width, height) != (record["width"], record["height"]):
            raise ValueError(f"Image dimensions changed since import: {source.name}")
        predictions = detector.predict(frame)
        boxes, scores, labels = _draft_boxes(predictions, width, height)
        suggestion = {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
            "model": str(model),
            "model_sha256": model_digest,
            "model_revision": revision,
            "raw_box_count": len(predictions),
            "overlap_threshold": 0.5,
            # Legacy manifest schema calls the fixed target class prompts.
            "prompts": ["cash"],
            "created_at": created,
            "confidence_threshold": config.detector.confidence,
            "image_size": config.detector.image_size,
        }
        saved = dataset.suggest(record["id"], suggestion)
        records.append(saved)
        # Save progress before starting another potentially expensive inference.
        _write_json(output / "predictions.json", {"version": 1, "records": records})
        logger.info(
            "%s/%s: %s — %s proposed boxes", index, len(pending), record["source"], len(boxes)
        )
    sheets = _contact_sheets(dataset, records, output)
    # Remove obsolete generated pages only after the new set renders successfully.
    # Partial human review can reduce the page count between saved-draft rebuilds.
    current_sheets = {Path(path) for path in sheets}
    for old_sheet in output.glob("contact-sheet-*.jpg"):
        if (
            old_sheet not in current_sheets
            and re.fullmatch(r"contact-sheet-[0-9]{2,}\.jpg", old_sheet.name)
            and old_sheet.is_file()
        ):
            old_sheet.unlink()
    identities = dict.fromkeys(
        (
            item["suggestion"].get("model"),
            item["suggestion"].get("model_sha256"),
            item["suggestion"].get("model_revision"),
        )
        for item in records
    )
    report = {
        "version": 1,
        "created_at": created,
        "dataset": str(dataset.directory),
        "engine": "yolo26m" if pending else "saved-drafts",
        # These fields describe this invocation; each saved draft retains its identity.
        "model": str(model) if pending else None,
        "model_sha256": model_digest,
        "model_revision": revision,
        "device": device,
        "new_frames": len(pending),
        "preserved_frames": preserved_frames,
        "models": [
            {"model": name, "model_sha256": sha, "model_revision": rev}
            for name, sha, rev in identities
        ],
        "confidence_threshold": config.detector.confidence,
        "image_size": config.detector.image_size,
        "frames": len(records),
        "frames_with_boxes": sum(bool(record["suggestion"]["boxes"]) for record in records),
        "boxes": sum(len(record["suggestion"]["boxes"]) for record in records),
        "raw_boxes": sum(
            record["suggestion"].get("raw_box_count", len(record["suggestion"]["boxes"]))
            for record in records
        ),
        "overlap_threshold": 0.5,
        "videos": sorted({record["source"].split(", ")[0] for record in records}),
        "contact_sheets": sheets,
        "review_note": (
            "AI drafts only. No-box frames are not negative labels. Review before training."
        ),
    }
    report_path = output / "report.json"
    _write_json(report_path, report)
    return report_path
