"""Generate cash-box drafts locally, leaving every result pending review."""

import hashlib
import logging
import math
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
        # Cash prompts describe the same class. Keep one proposal per overlapping object.
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


def auto_label(
    config: AppConfig,
    directory: Path,
    *,
    output: Path | None = None,
    engine: str = "world",
    grounding_model: str = "tiny",
    reference: Path | None = None,
) -> Path:
    """Persist model suggestions without accepting any positives or negatives."""
    if engine not in {"world", "grounding", "yoloe"}:
        raise ValueError("Auto-label engine must be world, grounding or yoloe")
    if engine == "world" and config.detector.backend != "world":
        raise ValueError("Auto-labeling needs the prompt model, not a previously trained detector")
    if grounding_model not in {"tiny", "base"}:
        raise ValueError("Grounding model must be tiny or base")
    if engine == "yoloe" and reference is None:
        raise ValueError("YOLOE labeling needs --reference with a boxed cash reference")
    dataset = Dataset(directory)
    pending = [record for record in dataset.records() if record["status"] == "unreviewed"]
    if not pending:
        raise ValueError("No unreviewed pictures. Import your videos before auto-labeling.")
    output = config.resolve("outputs/video-autolabel") if output is None else Path(output).resolve()
    if output.is_relative_to(dataset.directory) or dataset.directory.is_relative_to(output):
        raise ValueError("Auto-label output must be separate from the source dataset")
    output.mkdir(parents=True, exist_ok=True)
    created = datetime.now(UTC).isoformat()
    import cv2

    if engine == "world":
        model = config.resolve(config.detector.model)
        if not model.is_file():
            raise ValueError("Prepare the prompt detector before auto-labeling (make prepare)")
        model_digest = hashlib.sha256(model.read_bytes()).hexdigest()
        detector = MoneyDetector(config)
    elif engine == "yoloe":
        from dmotion.teachers import ReferenceTeacher

        model = config.resolve(config.detector.model)
        model_digest = hashlib.sha256(model.read_bytes()).hexdigest()
        detector = ReferenceTeacher(
            config.root,
            model,
            Path(reference).resolve(),
            confidence=config.detector.confidence,
            image_size=config.detector.image_size,
            device=config.detector.device,
        )
    else:
        from dmotion.grounding import GroundingMoneyDetector
        from dmotion.teachers import GROUNDING_MODELS

        model = GROUNDING_MODELS[grounding_model]
        model_digest = None
        detector = GroundingMoneyDetector(
            config.root,
            confidence=config.detector.confidence,
            prompts=config.detector.prompts,
            image_size=config.detector.image_size,
            device=config.detector.device,
            model_id=model,
        )
    records = []
    logger.info("Suggesting cash boxes for %s pictures on %s", len(pending), detector.device)
    revision = getattr(getattr(detector.model, "config", None), "_commit_hash", None)
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
            "reference": getattr(detector, "reference_provenance", None),
            "raw_box_count": len(predictions),
            "overlap_threshold": 0.5,
            "prompts": list(config.detector.prompts),
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
    report = {
        "version": 1,
        "created_at": created,
        "dataset": str(dataset.directory),
        "engine": engine,
        "model": str(model),
        "model_sha256": model_digest,
        "model_revision": revision,
        "device": detector.device,
        "prompts": list(config.detector.prompts),
        "confidence_threshold": config.detector.confidence,
        "image_size": config.detector.image_size,
        "frames": len(records),
        "frames_with_boxes": sum(bool(record["suggestion"]["boxes"]) for record in records),
        "boxes": sum(len(record["suggestion"]["boxes"]) for record in records),
        "raw_boxes": sum(record["suggestion"]["raw_box_count"] for record in records),
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
