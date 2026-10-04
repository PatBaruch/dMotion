"""Frozen session splits, validation calibration and candidate-only model comparison."""

import hashlib
import json
import math
import shutil
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from dmotion.autolabel import _overlap
from dmotion.config import AppConfig
from dmotion.dataset import Dataset, _write_json
from dmotion.detector import MoneyDetector
from dmotion.training import train_model

DEFAULT_THRESHOLDS = (0.1, 0.175, 0.25, 0.35, 0.5, 0.65, 0.8)


def score_predictions(records: list[dict], predictions: dict, threshold: float) -> dict:
    """Greedy one-to-one IoU>=0.5 matching; a person-sized cash box is a false positive."""
    tp = fp = fn = negative_frames = false_alarm_frames = positive_frames = found_frames = 0
    for record in records:
        detected = sorted(
            (p for p in predictions[record["id"]] if p["confidence"] >= threshold),
            key=lambda p: p["confidence"],
            reverse=True,
        )
        unmatched = set(range(len(record["boxes"])))
        matches = 0
        for prediction in detected:
            best = max(
                unmatched,
                key=lambda i: _overlap(prediction["box"], record["boxes"][i]),
                default=None,
            )
            if best is not None and _overlap(prediction["box"], record["boxes"][best]) >= 0.5:
                unmatched.remove(best)
                tp += 1
                matches += 1
            else:
                fp += 1
        fn += len(unmatched)
        if record["status"] == "negative":
            negative_frames += 1
            false_alarm_frames += bool(detected)
        else:
            positive_frames += 1
            found_frames += matches > 0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": threshold,
        "images": len(records),
        "true_positive_boxes": tp,
        "false_positive_boxes": fp,
        "missed_boxes": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "positive_frames": positive_frames,
        "cash_frames_found": found_frames,
        "negative_frames": negative_frames,
        "false_alarm_frames": false_alarm_frames,
        "false_alarm_rate": false_alarm_frames / negative_frames if negative_frames else None,
    }


def passes_gate(
    metrics: dict, *, max_false_alarm_rate: float, min_precision: float, min_recall: float
) -> bool:
    return (
        metrics["negative_frames"] > 0
        and metrics["positive_frames"] > 0
        and metrics["false_alarm_rate"] <= max_false_alarm_rate
        and metrics["precision"] >= min_precision
        and metrics["recall"] >= min_recall
    )


def calibrate(
    records: list[dict],
    predictions: dict,
    *,
    thresholds=DEFAULT_THRESHOLDS,
    max_false_alarm_rate: float = 0.05,
    min_precision: float = 0.8,
    min_recall: float = 0.5,
) -> dict:
    if not thresholds or any(not math.isfinite(t) or not 0 < t <= 1 for t in thresholds):
        raise ValueError("Calibration thresholds must be finite values in (0, 1]")
    scores = [score_predictions(records, predictions, t) for t in thresholds]
    passing = [
        s
        for s in scores
        if passes_gate(
            s,
            max_false_alarm_rate=max_false_alarm_rate,
            min_precision=min_precision,
            min_recall=min_recall,
        )
    ]
    best = max(passing or scores, key=lambda s: (s["f1"], s["recall"], s["threshold"]))
    return {"selected": best, "gate_passed": bool(passing), "thresholds": scores}


def _predict_records(detector, dataset: Dataset, records: list[dict]) -> tuple[dict, dict]:
    import cv2

    predictions, timings = {}, []
    for record in records:
        source = dataset.image_path(record)
        if hashlib.sha256(source.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Image changed: {source}")
        frame = cv2.imread(str(source))
        if frame is None or frame.shape[:2] != (record["height"], record["width"]):
            raise ValueError(f"Cannot decode recorded image dimensions: {source}")
        start = perf_counter()
        predictions[record["id"]] = [
            {"box": list(d.box), "confidence": d.confidence, "label": d.label}
            for d in detector.predict(frame)
        ]
        timings.append((perf_counter() - start) * 1000)
    return predictions, {
        "mean_ms": sum(timings) / len(timings) if timings else 0.0,
        "device": detector.device,
    }


def compare_detector(
    config: AppConfig,
    dataset: Dataset,
    splits: dict[str, str],
    model: Path,
    output: Path,
    *,
    max_false_alarm_rate=0.05,
    min_precision=0.8,
    min_recall=0.5,
) -> dict:
    records = [r for r in dataset.records() if r["status"] in {"positive", "negative"}]
    validation = [r for r in records if splits[r["group"]] == "val"]
    test = [r for r in records if splits[r["group"]] == "test"]
    settings = replace(
        config,
        detector=replace(
            config.detector,
            backend="trained",
            model=str(model),
            confidence=min(DEFAULT_THRESHOLDS),
            image_size=640,
        ),
    )
    detector = MoneyDetector(settings)
    val_predictions, val_timing = _predict_records(detector, dataset, validation)
    selection = calibrate(
        validation,
        val_predictions,
        max_false_alarm_rate=max_false_alarm_rate,
        min_precision=min_precision,
        min_recall=min_recall,
    )
    # Freeze the threshold before reading any test predictions.
    _write_json(output / "calibration.json", selection)
    test_predictions, test_timing = _predict_records(detector, dataset, test)
    measured = score_predictions(test, test_predictions, selection["selected"]["threshold"])
    report = {
        "model": str(model),
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "validation": selection,
        "test": measured,
        "validation_timing": val_timing,
        "test_timing": test_timing,
        "test_gate_passed": passes_gate(
            measured,
            max_false_alarm_rate=max_false_alarm_rate,
            min_precision=min_precision,
            min_recall=min_recall,
        ),
    }
    _write_json(
        output / "predictions.json", {"validation": val_predictions, "test": test_predictions}
    )
    _write_json(output / "report.json", report)
    return report


def run_harness(
    config: AppConfig,
    directory: Path,
    split_file: Path,
    *,
    output: Path | None = None,
    baseline: Path | None = None,
    epochs: int = 30,
    patience: int = 10,
    image_size: int = 640,
    device: str = "auto",
    max_false_alarm_rate: float = 0.05,
    min_precision: float = 0.8,
    min_recall: float = 0.5,
    initial_model: Path | None = None,
) -> Path:
    """Train from an immutable snapshot; never replace the active camera checkpoint."""
    for value in (max_false_alarm_rate, min_precision, min_recall):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Quality gate limits must be finite values between 0 and 1")
    source = Dataset(directory)
    records = source.records()
    if any(r["status"] == "unreviewed" for r in records):
        raise ValueError("Harness input contains unreviewed images. Review or exclude them first.")
    splits = json.loads(Path(split_file).read_text())
    reviewed = [r for r in records if r["status"] in {"positive", "negative"}]
    groups = {r["group"] for r in reviewed}
    if (
        not isinstance(splits, dict)
        or set(splits) != groups
        or any(s not in {"train", "val", "test"} for s in splits.values())
    ):
        raise ValueError(
            "Split plan must assign every reviewed group exactly once to train/val/test"
        )
    for split in ("train", "val", "test"):
        for status in ("positive", "negative"):
            if not any(r["status"] == status and splits[r["group"]] == split for r in reviewed):
                raise ValueError(f"Harness split {split} needs reviewed {status} examples")
    baseline = (
        config.resolve("models/money-spread.pt") if baseline is None else Path(baseline).resolve()
    )
    if not baseline.is_file():
        raise ValueError(f"Baseline checkpoint does not exist: {baseline}")
    if initial_model is not None:
        initial_model = Path(initial_model).expanduser().resolve()
        if not initial_model.is_file():
            raise ValueError(f"Initial model checkpoint does not exist: {initial_model}")
    name = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:6]
    output = config.resolve(f"outputs/harness/{name}") if output is None else Path(output).resolve()
    if output.is_relative_to(source.directory) or source.directory.is_relative_to(output):
        raise ValueError("Harness output must be separate from the dataset")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Harness output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    snapshot = Dataset(output / "dataset")
    with source._manifest_lock:
        # Read again under the lock so the frozen manifest and copied files match.
        if source.records() != records:
            raise ValueError(
                "Dataset changed while starting the run; retry with stable reviewed labels"
            )
        for record in reviewed:
            image = source.image_path(record)
            if hashlib.sha256(image.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"Image changed: {image}")
            target = snapshot.directory / record["image"]
            shutil.copyfile(image, target)
            if hashlib.sha256(target.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"Image changed during snapshot copy: {image}")
        _write_json(snapshot.manifest_path, {"version": 1, "records": reviewed})
    _write_json(output / "splits.json", splits)
    shutil.copyfile(baseline, output / "baseline.pt")
    initial_snapshot = None
    if initial_model is not None:
        initial_snapshot = output / "initial.pt"
        shutil.copyfile(
            output / "baseline.pt" if initial_model == baseline else initial_model, initial_snapshot
        )
    provenance = {
        "version": 1,
        "source_dataset": str(source.directory),
        "manifest_sha256": hashlib.sha256(snapshot.manifest_path.read_bytes()).hexdigest(),
        "group_splits": splits,
        "active_model_replaced": False,
        "initial_model_override": str(initial_model) if initial_model is not None else None,
        "initial_model_data_overlap_unknown": initial_model is not None,
        "note": "AI drafts must be reviewed. Legacy labels may lack reviewer attribution. "
        "A historical baseline may already have seen these evaluation recordings.",
    }
    _write_json(output / "run.json", {**provenance, "stage": "training"})
    candidate = train_model(
        config,
        snapshot.directory,
        epochs=epochs,
        patience=patience,
        image_size=image_size,
        device=device,
        candidate_directory=output / "model",
        group_splits=splits,
        evaluate_test=False,
        initial_model=initial_snapshot,
    )
    _write_json(
        output / "run.json", {**provenance, "stage": "evaluation", "candidate": str(candidate)}
    )
    gate = {
        "max_false_alarm_rate": max_false_alarm_rate,
        "min_precision": min_precision,
        "min_recall": min_recall,
    }
    eval_config = replace(config, detector=replace(config.detector, device=device))
    reports = {
        label: compare_detector(eval_config, snapshot, splits, model, output / label, **gate)
        for label, model in (("baseline", output / "baseline.pt"), ("candidate", candidate))
    }
    eligible = (
        reports["candidate"]["validation"]["gate_passed"]
        and reports["candidate"]["test_gate_passed"]
    )
    report = {
        **provenance,
        "stage": "complete",
        "candidate": str(candidate),
        "quality_gate": gate,
        "quality_gate_passed": eligible,
        "promotion_eligible": eligible and initial_model is None,
        "models": reports,
        "evaluation_note": "Thresholds selected on validation only. Test groups were not used "
        "to fit this candidate in this run. Initial custom checkpoints may have seen "
        "these groups earlier. Further iterations need fresh test recordings.",
        "test_command": f"dmotion run --mode trained --model {candidate} --confidence "
        f"{reports['candidate']['validation']['selected']['threshold']}",
    }
    _write_json(output / "report.json", report)
    _write_json(
        output / "run.json", {**provenance, "stage": "complete", "candidate": str(candidate)}
    )
    return output / "report.json"


def compare_label_drafts(
    directory: Path, prediction_files: list[Path], output: Path, *, confidence: float = 0.25
) -> Path:
    """Compare teacher drafts on exactly the same subsequently reviewed frames."""
    from dmotion.dataset import _validate_suggestion

    if not math.isfinite(confidence) or not 0 < confidence <= 1:
        raise ValueError("Comparison confidence must be in (0, 1]")
    if len(prediction_files) < 2:
        raise ValueError("Compare at least two labeling prediction files")
    dataset = Dataset(directory)
    truth = {r["id"]: r for r in dataset.records()}
    drafts, identities = [], None
    for file in prediction_files:
        rows = json.loads(Path(file).read_text())["records"]
        ids = {r["id"] for r in rows}
        if not ids or len(ids) != len(rows) or not ids <= truth.keys():
            raise ValueError("Draft predictions need unique IDs belonging to this dataset")
        if identities is not None and ids != identities:
            raise ValueError("Labeling teachers must be compared on the same frame IDs")
        identities = ids
        predicted = {}
        for row in rows:
            record = truth[row["id"]]
            if row["sha256"] != record["sha256"]:
                raise ValueError("Draft image provenance does not match the reviewed image")
            if record["status"] == "unreviewed":
                raise ValueError("Review or exclude every draft frame before comparing teachers")
            _validate_suggestion(row["suggestion"], record["width"], record["height"])
            predicted[row["id"]] = [
                {"box": box, "confidence": score}
                for box, score in zip(
                    row["suggestion"]["boxes"], row["suggestion"]["scores"], strict=True
                )
            ]
        drafts.append((Path(file).resolve(), rows[0]["suggestion"], predicted))
    records = [r for r in truth.values() if r["id"] in identities and r["status"] != "excluded"]
    if not records:
        raise ValueError("No reviewed frames to compare")
    for record in records:
        if hashlib.sha256(dataset.image_path(record).read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Reviewed image changed since import")
    output = Path(output).resolve()
    if output.is_relative_to(dataset.directory) or output in [f for f, _, _ in drafts]:
        raise ValueError("Comparison output must be separate from dataset and prediction inputs")
    _write_json(
        output,
        {
            "version": 1,
            "dataset": str(dataset.directory),
            "confidence": confidence,
            "note": "Teacher comparison against reviewed labels, not a trained-detector test. "
            "Review accuracy depends on the reviewers; "
            "AI-assisted review is not human ground truth.",
            "frame_ids": [r["id"] for r in records],
            "teachers": [
                {
                    "predictions": str(file),
                    "model": suggestion["model"],
                    "model_revision": suggestion.get("model_revision"),
                    "model_sha256": suggestion.get("model_sha256"),
                    "reference": suggestion.get("reference"),
                    "inference_confidence": suggestion.get("confidence_threshold"),
                    "metrics": score_predictions(records, predictions, confidence),
                }
                for file, suggestion, predictions in drafts
            ],
        },
    )
    return output
