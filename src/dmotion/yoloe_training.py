"""Reviewed-data snapshots and candidate-only YOLOE Small detection fine-tuning."""

import hashlib
import json
import logging
import math
import re
import shutil
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from dmotion.config import AppConfig
from dmotion.dataset import Dataset, _write_json, export_dataset
from dmotion.detector import prepare_environment, select_device
from dmotion.training import _numeric_metrics

logger = logging.getLogger(__name__)
MODEL_ID = "yoloe-26s-seg.pt"
ARCHITECTURE = "yoloe-26s.yaml"


def snapshot_reviewed(directories: list[Path], output: Path) -> Dataset:
    """Deduplicate bytes while preserving labels, recording groups and source provenance."""
    if not directories:
        raise ValueError("Provide at least one reviewed dataset")
    unique = {}
    for directory in directories:
        if not (Path(directory) / "manifest.json").is_file():
            raise ValueError(f"Dataset manifest is missing: {directory}/manifest.json")
        source = Dataset(directory)
        with source._manifest_lock:
            records = source.records()
            if any(record["status"] == "unreviewed" for record in records):
                raise ValueError(f"Review or skip all unreviewed images in {source.directory}")
            for record in records:
                if record["status"] not in {"positive", "negative"}:
                    continue
                image = source.image_path(record)
                contents = image.read_bytes()
                digest = hashlib.sha256(contents).hexdigest()
                if digest != record["sha256"]:
                    raise ValueError(f"Image changed since review: {image}")
                origin = {"dataset": str(source.directory), "record_id": record["id"]}
                if digest in unique:
                    previous = unique[digest][0]
                    if any(
                        previous[key] != record[key]
                        for key in ("status", "boxes", "group", "width", "height")
                    ):
                        raise ValueError(
                            f"Conflicting reviews or groups for duplicate image {digest}"
                        )
                    previous["origins"].append(origin)
                    continue
                copy = {**record, "id": digest, "origins": [origin]}
                copy["image"] = f"images/{digest}{image.suffix.lower()}"
                unique[digest] = (copy, contents)
    if not unique:
        raise ValueError("No reviewed cash or no-cash examples")
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise ValueError("The snapshot destination must be new")
    snapshot = Dataset(output)
    for record, contents in unique.values():
        (snapshot.directory / record["image"]).write_bytes(contents)
    _write_json(snapshot.manifest_path, {"version": 1, "records": [v[0] for v in unique.values()]})
    return snapshot


def head_freeze(model) -> tuple[list[str], str]:
    """Freeze features, box regression and prompt modules; retain terminal class convolutions."""
    modules = model.model.model
    head_index = len(modules) - 1
    freeze = [str(index) for index in range(head_index)]
    for name, _child in modules[-1].named_children():
        if name in {"cv3", "one2one_cv3"}:
            freeze.extend(f"{head_index}.{name}.{i}.{j}" for i in range(3) for j in (0, 1))
        else:
            freeze.append(f"{head_index}.{name}")
    allowed = rf"model\.{head_index}\.(?:one2one_)?cv3\.[0-2]\.2\."
    return freeze, allowed


def train_yoloe(
    config: AppConfig,
    directories: list[Path],
    *,
    output: Path | None = None,
    pretrained: Path | None = None,
    split_file: Path | None = None,
    epochs: int = 30,
    patience: int = 8,
    image_size: int = 640,
    device: str = "auto",
) -> Path:
    """Train a fixed-class cash candidate, leaving active and reference models untouched."""
    if type(epochs) is not int or not 1 <= epochs <= 500:
        raise ValueError("Epochs must be an integer between 1 and 500")
    if type(patience) is not int or not 0 <= patience <= 500:
        raise ValueError("Patience must be an integer between 0 and 500")
    if type(image_size) is not int or image_size < 32 or image_size % 32:
        raise ValueError("Image size must be a positive multiple of 32")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("Device must be auto, cpu, mps or a CUDA device")
    weights = config.resolve(f"models/{MODEL_ID}") if pretrained is None else pretrained.resolve()
    if not weights.is_file():
        raise ValueError(f"Prepare the selected YOLOE Small checkpoint first: {weights}")
    if weights.name != MODEL_ID:
        raise ValueError(f"This training route requires the selected {MODEL_ID} checkpoint")
    splits = None if split_file is None else json.loads(split_file.read_text())
    timestamp = datetime.now(UTC)
    name = timestamp.strftime("yoloe-%Y%m%d-%H%M%S-") + uuid4().hex[:6]
    parent = config.resolve("outputs/yoloe-training") if output is None else output.resolve()
    run = parent / name
    snapshot = snapshot_reviewed(directories, run / "dataset")
    data = export_dataset(snapshot, run / "yolo", class_name="cash", group_splits=splits)
    provenance = json.loads((data.parent / "export-report.json").read_text())
    for split in ("train", "val", "test"):
        if not provenance["splits"][split]["negative"]:
            raise ValueError(f"{split} needs reviewed no-cash examples to measure false positives")
    initial = run / MODEL_ID
    shutil.copyfile(weights, initial)
    digest = hashlib.sha256(initial.read_bytes()).hexdigest()
    info = {
        "version": 1,
        "created_at": timestamp.isoformat(),
        "run": name,
        "model_id": MODEL_ID,
        "model_revision": f"sha256:{digest}",
        "pretrained": str(weights),
        "pretrained_sha256": digest,
        "architecture": ARCHITECTURE,
        "class": "cash",
        "source_datasets": [str(Path(p).resolve()) for p in directories],
        "dataset": str(data),
        "snapshot_manifest_sha256": hashlib.sha256(snapshot.manifest_path.read_bytes()).hexdigest(),
        "splits": provenance["splits"],
        "group_splits": provenance["group_splits"],
        "requested_epochs": epochs,
        "patience": patience,
        "image_size": image_size,
        "candidate_only": True,
        "active_model_replaced": False,
        "note": "Fixed cash classes; reference-photo prompting is not retained. "
        "Reviewed status preserves legacy provenance, not an independent label audit. "
        "Calibrate confidence on validation before evaluating test or promoting the candidate.",
    }
    _write_json(run / "training-info.json", {**info, "stage": "preparing"})
    prepare_environment(config.root)
    import torch
    from ultralytics import YOLO, YOLOE, settings
    from ultralytics.models.yolo.yoloe import YOLOEPETrainer

    settings.update({"sync": False})
    selected_device = select_device(device, torch)
    info.update(device=selected_device, ultralytics_version=version("ultralytics"))
    model = YOLOE(ARCHITECTURE).load(str(initial))
    freeze, allowed = head_freeze(model)
    updates = {"optimizer_steps": 0, "positive_lr_optimizer_steps": 0}
    hooks = []

    def count_step(optimizer, _arguments, _keywords):
        updates["optimizer_steps"] += 1
        if any(
            math.isfinite(float(group["lr"])) and float(group["lr"]) > 0
            for group in optimizer.param_groups
        ):
            updates["positive_lr_optimizer_steps"] += 1

    def audit_trainable(trainer):
        parameters = [(key, p) for key, p in trainer.model.named_parameters() if p.requires_grad]
        if not parameters or any(not re.match(allowed, key) for key, _p in parameters):
            raise RuntimeError("YOLOE head-only training has unexpected trainable parameters")
        info["trainable_parameters"] = [key for key, _p in parameters]
        info["trainable_parameter_count"] = sum(p.numel() for _key, p in parameters)
        hooks.append(trainer.optimizer.register_step_post_hook(count_step))
        _write_json(run / "training-info.json", {**info, "stage": "training"})

    model.add_callback("on_train_start", audit_trainable)
    try:
        model.train(
            data=str(data),
            trainer=YOLOEPETrainer,
            freeze=freeze,
            epochs=epochs,
            patience=patience,
            imgsz=image_size,
            device=selected_device,
            batch=4,
            nbs=4,
            workers=0,
            seed=42,
            amp=False,
            plots=False,
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.1,
            warmup_epochs=1.0,
            mosaic=0.0,
            project=str(run),
            name="fit",
            exist_ok=False,
        )
        if not updates["positive_lr_optimizer_steps"]:
            raise RuntimeError("No positive-learning-rate optimizer updates; no candidate saved")
        best = Path(model.trainer.best)
        if not best.is_file():
            raise RuntimeError("YOLOE training finished without a best checkpoint")
        evaluation = YOLO(str(best)).val(
            data=str(data),
            split="val",
            imgsz=image_size,
            device=selected_device,
            batch=4,
            workers=0,
            plots=False,
            verbose=False,
            project=str(run),
            name="validation",
            exist_ok=False,
        )
        candidate = run / "candidate.pt"
        shutil.copyfile(best, candidate)
        info.update(
            stage="complete",
            model=str(candidate),
            checkpoint=str(best),
            model_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
            completed_epochs=int(model.trainer.epoch) + 1,
            validation_metrics=_numeric_metrics(evaluation),
            **updates,
        )
        _write_json(candidate.with_suffix(".json"), info)
        _write_json(run / "training-info.json", info)
        logger.info("YOLOE candidate saved: %s", candidate)
        return candidate
    except Exception as exc:
        _write_json(
            run / "training-info.json", {**info, **updates, "stage": "failed", "error": str(exc)}
        )
        raise
    finally:
        for hook in hooks:
            hook.remove()
