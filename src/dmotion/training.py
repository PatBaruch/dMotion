"""Bounded local fine-tuning, followed by evaluation on untouched session groups."""

import hashlib
import json
import logging
import math
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from dmotion.config import AppConfig
from dmotion.dataset import Dataset, _write_json, export_dataset
from dmotion.detector import prepare_environment, select_device

logger = logging.getLogger(__name__)


def _numeric_metrics(result) -> dict[str, float]:
    metrics = {}
    for key, value in getattr(result, "results_dict", {}).items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            metrics[str(key)] = numeric
    return metrics


def train_model(
    config: AppConfig,
    dataset_directory: Path,
    *,
    epochs: int = 30,
    image_size: int = 640,
    device: str = "auto",
) -> Path:
    if type(epochs) is not int or not 1 <= epochs <= 500:
        raise ValueError("Training epochs must be an integer between 1 and 500")
    if type(image_size) is not int or image_size < 32 or image_size % 32:
        raise ValueError("Training image size must be a positive multiple of 32")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("Training device must be auto, cpu, mps, or a CUDA device")
    dataset = Dataset(dataset_directory)
    data_path = export_dataset(dataset, config.root / "data" / "yolo")
    export_report = json.loads((data_path.parent / "export-report.json").read_text())
    for warning in export_report["warnings"]:
        logger.warning(warning)

    prepare_environment(config.root)
    import torch
    from ultralytics import YOLO, settings

    settings.update({"sync": False})
    selected_device = select_device(device, torch)
    timestamp = datetime.now(UTC)
    run_name = timestamp.strftime("spread-%Y%m%d-%H%M%S-") + uuid4().hex[:6]
    project = config.root / "outputs" / "training"
    pretrained_path = config.root / "models" / "yolo26n.pt"
    pretrained_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Training reviewed spreads on %s; run %s", selected_device, run_name)
    model = YOLO(str(pretrained_path))
    updates = {"optimizer_steps": 0, "positive_lr_optimizer_steps": 0}
    step_hooks = []

    def count_optimizer_step(optimizer, _arguments, _keywords):
        updates["optimizer_steps"] += 1
        if any(
            math.isfinite(float(group["lr"])) and float(group["lr"]) > 0
            for group in optimizer.param_groups
        ):
            updates["positive_lr_optimizer_steps"] += 1

    def install_step_counter(trainer):
        # A post-hook observes actual updates, including whether AMP skipped a step.
        step_hooks.append(trainer.optimizer.register_step_post_hook(count_optimizer_step))

    model.add_callback("on_train_start", install_step_counter)
    try:
        model.train(
            data=str(data_path),
            epochs=epochs,
            imgsz=image_size,
            device=selected_device,
            batch=4,
            nbs=4,  # Avoid accumulating many epochs before an update on small datasets.
            workers=0,
            patience=10,
            pretrained=True,
            seed=42,
            project=str(project),
            name=run_name,
            exist_ok=False,
            plots=False,
        )
    finally:
        for handle in step_hooks:
            handle.remove()
    if not updates["positive_lr_optimizer_steps"]:
        raise RuntimeError(
            "Training produced no optimizer updates at a positive learning rate; "
            "model was not saved"
        )
    logger.info(
        "Training applied %s optimizer steps, %s at a positive learning rate",
        updates["optimizer_steps"],
        updates["positive_lr_optimizer_steps"],
    )
    best = Path(model.trainer.best)
    if not best.is_file():
        raise RuntimeError("Training finished without a best model checkpoint")
    # Evaluation never uses the training or validation sessions.
    evaluation = YOLO(str(best)).val(
        data=str(data_path),
        split="test",
        imgsz=image_size,
        device=selected_device,
        batch=4,
        workers=0,
        project=str(project),
        name=f"{run_name}-test",
        exist_ok=False,
        plots=False,
        verbose=False,
    )
    destination = config.root / "models" / "money-spread.pt"
    descriptor, temporary = tempfile.mkstemp(prefix=".money-spread-", dir=destination.parent)
    os.close(descriptor)
    try:
        shutil.copyfile(best, temporary)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    info = {
        "version": 1,
        "created_at": timestamp.isoformat(),
        "run": run_name,
        "pretrained": str(pretrained_path),
        "model": str(destination),
        "model_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "checkpoint": str(best),
        "dataset": str(data_path),
        "requested_epochs": epochs,
        "completed_epochs": int(model.trainer.epoch) + 1,
        **updates,
        "image_size": image_size,
        "device": selected_device,
        "splits": export_report["splits"],
        "warnings": export_report["warnings"],
        "test_metrics": _numeric_metrics(evaluation),
        "evaluation_note": "Test images come from groups not used to fit or select this model.",
    }
    _write_json(project / run_name / "dataset-provenance.json", export_report)
    _write_json(project / run_name / "training-info.json", info)
    _write_json(destination.with_suffix(".json"), info)
    logger.info("Trained model saved to %s; test metrics: %s", destination, info["test_metrics"])
    return destination
