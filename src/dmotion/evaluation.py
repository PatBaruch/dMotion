"""Compare cash checkpoints with session splits and validation-only calibration.

The scoring/calibration approach is adapted from the session-safe harness in
PR #5. Evaluation does not train, change labels, or replace the live checkpoint.
"""

import hashlib
import json
import math
from contextlib import ExitStack
from dataclasses import replace
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from uuid import uuid4

from dmotion.autolabel import _overlap
from dmotion.checkpoint import checkpoint_snapshot
from dmotion.config import AppConfig, validate
from dmotion.dataset import Dataset, _write_json
from dmotion.detector import MoneyDetector

DEFAULT_THRESHOLDS = tuple(value / 100 for value in range(5, 100, 5))
DEFAULT_GATE = {"max_false_alarm_rate": 0.05, "min_precision": 0.8, "min_recall": 0.5}


def _rate(value: float, name: str, *, minimum: float = 0.0) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not minimum <= value <= 1
    ):
        raise ValueError(f"{name} must be a finite number between {minimum} and 1")


def score_predictions(
    records: list[dict], predictions: dict, threshold: float, *, iou_threshold: float = 0.5
) -> dict:
    """Match boxes once, in descending confidence order, using original pixel IoU."""
    _rate(threshold, "Confidence threshold", minimum=0.000001)
    _rate(iou_threshold, "IoU threshold", minimum=0.000001)
    tp = fp = fn = negative = alarms = positive = found = 0
    for record in records:
        if record["status"] not in {"positive", "negative"}:
            raise ValueError("Scoring requires explicitly reviewed positive/negative records")
        detected = []
        for prediction in predictions[record["id"]]:
            _rate(prediction["confidence"], "Prediction confidence")
            box = prediction["box"]
            if (
                len(box) != 4
                or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in box)
                or not 0 <= box[0] < box[2] <= record["width"]
                or not 0 <= box[1] < box[3] <= record["height"]
            ):
                raise ValueError("Predicted boxes must have positive area inside the image")
            if prediction["confidence"] >= threshold:
                detected.append(prediction)
        detected.sort(key=lambda p: p["confidence"], reverse=True)
        unmatched = set(range(len(record["boxes"])))
        matches = 0
        for prediction in detected:
            best = max(
                sorted(unmatched),
                key=lambda i: _overlap(prediction["box"], record["boxes"][i]),
                default=None,
            )
            if (
                best is not None
                and _overlap(prediction["box"], record["boxes"][best]) >= iou_threshold
            ):
                unmatched.remove(best)
                tp += 1
                matches += 1
            else:
                fp += 1
        fn += len(unmatched)
        if record["status"] == "negative":
            negative += 1
            alarms += bool(detected)
        else:
            positive += 1
            found += matches > 0
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
        "positive_frames": positive,
        "cash_frames_found": found,
        "negative_frames": negative,
        "false_alarm_frames": alarms,
        "false_alarm_rate": alarms / negative if negative else None,
    }


def gate_failures(metrics: dict, gate: dict) -> list[str]:
    failures = []
    if not metrics["positive_frames"]:
        failures.append("No positive frames")
    if not metrics["negative_frames"]:
        failures.append("No negative frames")
    elif metrics["false_alarm_rate"] > gate["max_false_alarm_rate"]:
        failures.append("False-alarm frame rate exceeds the maximum")
    if metrics["precision"] < gate["min_precision"]:
        failures.append("Box precision is below the minimum")
    if metrics["recall"] < gate["min_recall"]:
        failures.append("Box recall is below the minimum")
    return failures


def calibrate(records: list[dict], predictions: dict, thresholds, gate: dict, iou: float) -> dict:
    scores = [score_predictions(records, predictions, t, iou_threshold=iou) for t in thresholds]
    passing = [score for score in scores if not gate_failures(score, gate)]
    selected = max(passing or scores, key=lambda s: (s["f1"], s["recall"], s["threshold"]))
    return {
        "selected": selected,
        "gate_passed": bool(passing),
        "failures": gate_failures(selected, gate),
        "thresholds": scores,
    }


def _split_plan(raw: dict, records: list[dict]) -> dict[str, str]:
    """Accept an explicit group map or an export tied to these exact reviewed records."""
    reviewed = [r for r in records if r["status"] in {"positive", "negative"}]
    if not isinstance(raw, dict):
        raise ValueError("Split plan must be a group map or dataset export report")
    if isinstance(raw.get("records"), list):
        exported = raw["records"]
        expected = {r["id"]: r for r in reviewed}
        if len(exported) != len(expected) or {r["id"] for r in exported} != set(expected):
            raise ValueError("Export records do not match the reviewed dataset")
        splits = {}
        for row in exported:
            if any(
                row[k] != expected[row["id"]][k] for k in ("sha256", "group", "status", "boxes")
            ):
                raise ValueError("Export labels/image provenance differ from the reviewed dataset")
            if row["group"] in splits and splits[row["group"]] != row["split"]:
                raise ValueError("A recording group appears in more than one split")
            splits[row["group"]] = row["split"]
    else:
        splits = raw
    if set(splits) != {r["group"] for r in reviewed} or any(
        not isinstance(s, str) or s not in {"train", "val", "test"} for s in splits.values()
    ):
        raise ValueError("Assign every reviewed group exactly once to train, val, or test")
    for split in ("val", "test"):
        for status in ("positive", "negative"):
            if not any(r["status"] == status and splits[r["group"]] == split for r in reviewed):
                raise ValueError(
                    f"{split} needs reviewed {status} examples from independent groups"
                )
    return splits


def _freeze_dataset(source: Dataset, output: Path, raw_splits: dict) -> tuple[Dataset, dict, dict]:
    with source.locked_records() as records:
        if any(r["status"] == "unreviewed" for r in records):
            raise ValueError("Review or exclude every unreviewed image before evaluation")
        splits = _split_plan(raw_splits, records)
        manifest = source.manifest_path.read_bytes()
        frozen = Dataset(output / "dataset")
        evaluated = [
            r
            for r in records
            if r["status"] in {"positive", "negative"} and splits[r["group"]] in {"val", "test"}
        ]
        for record in evaluated:
            contents = source.image_path(record).read_bytes()
            if hashlib.sha256(contents).hexdigest() != record["sha256"]:
                raise ValueError(f"Image changed since review: {record['id']}")
            target = frozen.image_path(record)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contents)
        _write_json(frozen.manifest_path, {"version": 1, "records": evaluated})
        (output / "source-manifest.json").write_bytes(manifest)
    _write_json(output / "splits.json", splits)
    provenance = {
        "source_dataset": str(source.directory),
        "source_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "evaluation_manifest_sha256": hashlib.sha256(frozen.manifest_path.read_bytes()).hexdigest(),
        "group_splits": splits,
        "excluded_frames": sum(r["status"] == "excluded" for r in records),
        "labels_note": "Reviewed statuses; reviewer identity and human accuracy are not verified.",
    }
    return frozen, splits, provenance


def _predict_records(detector, dataset: Dataset, records: list[dict], *, warmup=False) -> tuple:
    import cv2
    import numpy as np

    predictions, timings, warmup_ms = {}, [], None
    for index, record in enumerate(records):
        contents = dataset.image_path(record).read_bytes()
        if hashlib.sha256(contents).hexdigest() != record["sha256"]:
            raise ValueError(f"Frozen evaluation image changed: {record['id']}")
        frame = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.shape[:2] != (record["height"], record["width"]):
            raise ValueError(f"Cannot decode recorded image dimensions: {record['id']}")
        if warmup and index == 0:
            start = perf_counter()
            detector.predict(frame)
            warmup_ms = (perf_counter() - start) * 1000
        start = perf_counter()
        detections = detector.predict(frame)
        timings.append((perf_counter() - start) * 1000)
        if any(d.label not in {"cash", "money_spread"} for d in detections):
            raise ValueError("Evaluation requires single-class cash predictions")
        predictions[record["id"]] = [
            {"box": list(d.box), "confidence": d.confidence, "label": d.label} for d in detections
        ]
    return predictions, {
        "images": len(timings),
        "mean_ms": mean(timings),
        "median_ms": median(timings),
        "p95_ms": sorted(timings)[math.ceil(len(timings) * 0.95) - 1],
        "warmup_ms": warmup_ms,
        "device": detector.device,
        "scope": "Predict wall time; decoding excluded; no webcam/audio measurement.",
    }


def _markdown(report: dict) -> str:
    lines = [
        "# YOLO26m checkpoint evaluation",
        "",
        f"Candidate verdict: **{report['verdict']}**",
        "",
        "Thresholds are selected on validation only. Metrics match boxes one-to-one at "
        f"IoU >= {report['iou_threshold']}. False alarms count negative frames, not audio alerts.",
        "",
        "| Model / split | Confidence | Box precision | Box recall | "
        "Cash frames | False-alarm frames | Gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    gate = report["quality_gate"]
    lines[3:3] = [
        f"Frame gates: box precision >= {gate['min_precision']:.0%}, "
        f"box recall >= {gate['min_recall']:.0%}, "
        f"false-alarm frame rate <= {gate['max_false_alarm_rate']:.0%}.",
        "",
    ]
    details = []
    for name, model in report["models"].items():
        for split in ("validation", "test"):
            result = model[split]
            if result is None:
                lines.append(f"| {name} / {split} | — | — | — | — | — | Not run |")
                continue
            score = result["selected"] if split == "validation" else result["metrics"]
            lines.append(
                f"| {name} / {split} | {score['threshold']:.3f} | {score['precision']:.1%} | "
                f"{score['recall']:.1%} | "
                f"{score['cash_frames_found']}/{score['positive_frames']} | "
                f"{score['false_alarm_frames']}/{score['negative_frames']} | "
                f"{'PASS' if result['gate_passed'] else 'FAIL'} |"
            )
            if result["failures"]:
                details.append(f"- {name} {split}: " + "; ".join(result["failures"]) + ".")
    lines.extend(["", *details])
    lines.extend(
        [
            "",
            "## Inference timing",
            "",
            "Prediction wall time in milliseconds; decoding and validation warm-up excluded.",
            "",
            "| Model / split | Mean | Median | p95 | Device |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for name, model in report["models"].items():
        timings = [("validation", model["validation_timing"])]
        if model["test"] is not None:
            timings.append(("test", model["test"]["timing"]))
        for split, timing in timings:
            lines.append(
                f"| {name} / {split} | {timing['mean_ms']:.1f} | "
                f"{timing['median_ms']:.1f} | {timing['p95_ms']:.1f} | {timing['device']} |"
            )
    lines.extend(["", "## Limits", "", *[f"- {note}" for note in report["limitations"]]])
    lines.extend(["", "## Checkpoint identity", ""])
    for name, model in report["models"].items():
        lines.append(f"- {name}: `{model['checkpoint']}`; `{model['model_revision']}`")
    return "\n".join(lines) + "\n"


def evaluate_checkpoints(
    config: AppConfig,
    directory: Path,
    split_file: Path,
    candidate: Path,
    *,
    baseline: Path | None = None,
    output: Path | None = None,
    thresholds=DEFAULT_THRESHOLDS,
    max_false_alarm_rate: float = 0.05,
    min_precision: float = 0.8,
    min_recall: float = 0.5,
    iou_threshold: float = 0.5,
    test_context: str = "previously-inspected",
) -> Path:
    """Write a fresh evaluation run; only a passing candidate validation unlocks test scoring."""
    validate(config)
    gate = {
        "max_false_alarm_rate": max_false_alarm_rate,
        "min_precision": min_precision,
        "min_recall": min_recall,
    }
    for name, value in gate.items():
        _rate(value, name)
    if not thresholds:
        raise ValueError("Provide at least one calibration threshold")
    for threshold in thresholds:
        _rate(threshold, "Calibration threshold", minimum=0.000001)
    thresholds = sorted(set(thresholds))
    _rate(iou_threshold, "IoU threshold", minimum=0.000001)
    if test_context not in {"fresh", "previously-inspected"}:
        raise ValueError("Test context must be fresh or previously-inspected")
    directory = Path(directory).expanduser().resolve()
    if not (directory / "manifest.json").is_file():
        raise ValueError("Evaluation requires an existing reviewed dataset manifest")
    source = Dataset(directory)
    split_file = Path(split_file).expanduser().resolve()
    split_bytes = split_file.read_bytes()
    raw_splits = json.loads(split_bytes)
    checkpoints = {
        "baseline": config.resolve(str(baseline or config.detector.model)),
        "candidate": config.resolve(str(candidate)),
    }
    if any(not model.is_file() for model in checkpoints.values()):
        raise ValueError("Baseline and candidate must be existing local cash checkpoints")
    name = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:6]
    output = config.resolve(str(output or f"outputs/evaluation/{name}"))
    if output.is_relative_to(directory) or directory.is_relative_to(output):
        raise ValueError("Evaluation output must be separate from the source dataset")
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "run.json", {"stage": "preparing", "active_model_replaced": False})
    try:
        dataset, splits, provenance = _freeze_dataset(source, output, raw_splits)
        provenance.update(
            split_file=str(split_file), split_file_sha256=hashlib.sha256(split_bytes).hexdigest()
        )
        records = dataset.records()
        validation = [r for r in records if splits[r["group"]] == "val"]
        test = [r for r in records if splits[r["group"]] == "test"]
        models, predictions, detectors = {}, {}, {}
        with ExitStack() as stack:
            snapshots = {
                label: stack.enter_context(checkpoint_snapshot(model, output / ".checkpoints"))
                for label, model in checkpoints.items()
            }
            for label, model in checkpoints.items():
                frozen, digest = snapshots[label]
                settings = replace(
                    config,
                    detector=replace(
                        config.detector, model=str(frozen), confidence=min(thresholds)
                    ),
                )
                detector = MoneyDetector(settings)
                if detector.model_sha256 != digest:
                    raise ValueError("Loaded model provenance differs from the frozen checkpoint")
                detectors[label] = detector
                val_predictions, timing = _predict_records(
                    detector, dataset, validation, warmup=True
                )
                selection = calibrate(validation, val_predictions, thresholds, gate, iou_threshold)
                predictions[label] = {"validation": val_predictions}
                models[label] = {
                    "checkpoint": str(model),
                    "model_id": model.name,
                    "model_revision": f"sha256:{digest}",
                    "model_sha256": digest,
                    "device": detector.device,
                    "validation": selection,
                    "validation_timing": timing,
                    "test": None,
                }
            # Durable receipt is written before either model produces test predictions.
            _write_json(
                output / "calibration.json",
                {
                    **provenance,
                    "quality_gate": gate,
                    "iou_threshold": iou_threshold,
                    "image_size": config.detector.image_size,
                    "models": models,
                },
            )
            if models["candidate"]["validation"]["gate_passed"]:
                _write_json(
                    output / "run.json", {"stage": "testing", "active_model_replaced": False}
                )
                for label, detector in detectors.items():
                    test_predictions, timing = _predict_records(detector, dataset, test)
                    predictions[label]["test"] = test_predictions
                    score = score_predictions(
                        test,
                        test_predictions,
                        models[label]["validation"]["selected"]["threshold"],
                        iou_threshold=iou_threshold,
                    )
                    failures = gate_failures(score, gate)
                    models[label]["test"] = {
                        "metrics": score,
                        "gate_passed": not failures,
                        "failures": failures,
                        "timing": timing,
                    }
        passed = (
            models["candidate"]["test"] is not None and models["candidate"]["test"]["gate_passed"]
        )
        verdict = (
            "passed_frame_gates"
            if passed
            else "failed_test"
            if models["candidate"]["test"] is not None
            else "failed_validation_test_not_run"
        )
        versions = {}
        for package in ("dmotion", "ultralytics", "torch", "opencv-python", "numpy"):
            try:
                versions[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                versions[package] = "not installed"
        report = {
            "version": 1,
            "created_at_utc": datetime.now(UTC).isoformat(),
            **provenance,
            "stage": "complete",
            "quality_gate": gate,
            "iou_threshold": iou_threshold,
            "image_size": config.detector.image_size,
            "runtime_versions": versions,
            "models": models,
            "candidate_gate_passed": passed,
            "verdict": verdict,
            "test_context_declared": test_context,
            "active_model_replaced": False,
            "limitations": [
                f"Declared test context: {test_context}; independence is not verified.",
                "Earlier checkpoint training overlap and label reviewer identity are not verified.",
                "Thresholds use validation only. Test is skipped when candidate validation fails.",
                "Inspected test sessions must not guide more tuning; use new independent sessions.",
                "Frame gates do not establish webcam/audio acceptance or authorize promotion.",
            ],
        }
        _write_json(output / "predictions.json", predictions)
        (output / "report.md").write_text(_markdown(report), encoding="utf-8")
        _write_json(
            output / "run.json",
            {"stage": "complete", "verdict": verdict, "active_model_replaced": False},
        )
        # Publish the canonical completed report last, after every other write succeeds.
        _write_json(output / "report.json", report)
        return output / "report.json"
    except Exception as error:
        _write_json(
            output / "run.json",
            {"stage": "failed", "error": str(error), "active_model_replaced": False},
        )
        raise
