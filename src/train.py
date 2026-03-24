"""Training entrypoint for config-driven cold-protein DTI experiments."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.config import ConfigError, load_and_resolve_config, save_config
from src.data.dataloader import build_dataloaders, describe_graph_cache, describe_splits
from src.metrics import build_metrics_payload
from src.utils import (
    build_logger,
    choose_device,
    environment_snapshot,
    prepare_run_directory,
    round_nested,
    set_global_seed,
    try_import_torch,
    write_json,
)


class NonFiniteValueError(RuntimeError):
    """Raised when training or evaluation encounters NaN/Inf values."""

    def __init__(
        self,
        *,
        stage: str,
        quantity: str,
        value: float | None = None,
        epoch: int | None = None,
        batch_index: int | None = None,
    ) -> None:
        location_parts = [f"stage={stage}", f"quantity={quantity}"]
        if epoch is not None:
            location_parts.append(f"epoch={epoch}")
        if batch_index is not None:
            location_parts.append(f"batch={batch_index}")
        if value is not None:
            location_parts.append(f"value={value}")
        super().__init__("Encountered non-finite numeric value (" + ", ".join(location_parts) + ").")
        self.stage = stage
        self.quantity = quantity
        self.value = value
        self.epoch = epoch
        self.batch_index = batch_index

    def to_payload(
        self,
        *,
        resume_checkpoint: Path | None,
        best_checkpoint: Path | None,
        best_metric: float | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "reason": "non_finite_value",
            "stage": self.stage,
            "quantity": self.quantity,
            "epoch": self.epoch,
            "batch_index": self.batch_index,
            "observed_value": self.value,
            "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
            "best_checkpoint": str(best_checkpoint) if best_checkpoint is not None else None,
        }
        if best_metric is not None and math.isfinite(best_metric):
            payload["best_metric"] = best_metric
        return payload


class BatchValidationError(ValueError):
    """Raised when a batch fails explicit BCE/logit/label validation checks."""


def _move_batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _build_grad_scaler(torch_module: Any, device: str, enabled: bool) -> Any:
    """Create an AMP GradScaler across old and new PyTorch AMP APIs."""
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    if hasattr(torch_module, "amp") and hasattr(torch_module.amp, "GradScaler"):
        return torch_module.amp.GradScaler(device=device_type, enabled=enabled)
    return torch_module.cuda.amp.GradScaler(enabled=enabled)


def _autocast_context(torch_module: Any, device: str, enabled: bool) -> Any:
    """Create an autocast context manager across old and new PyTorch AMP APIs."""
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    if hasattr(torch_module, "amp") and hasattr(torch_module.amp, "autocast"):
        return torch_module.amp.autocast(device_type=device_type, enabled=enabled)
    return torch_module.cuda.amp.autocast(enabled=enabled)


def _coerce_finite_float(
    value: Any,
    *,
    stage: str,
    quantity: str,
    epoch: int | None = None,
    batch_index: int | None = None,
) -> float:
    scalar = float(value)
    if not math.isfinite(scalar):
        raise NonFiniteValueError(
            stage=stage,
            quantity=quantity,
            value=scalar,
            epoch=epoch,
            batch_index=batch_index,
        )
    return scalar


def _ensure_finite_tensor(
    torch_module: Any,
    tensor: Any,
    *,
    stage: str,
    quantity: str,
    epoch: int | None = None,
    batch_index: int | None = None,
) -> None:
    if not bool(torch_module.isfinite(tensor).all().item()):
        raise NonFiniteValueError(
            stage=stage,
            quantity=quantity,
            epoch=epoch,
            batch_index=batch_index,
        )


def _validate_binary_loss_inputs(
    torch_module: Any,
    logits: Any,
    labels: Any,
    *,
    stage: str,
    epoch: int | None = None,
    batch_index: int | None = None,
) -> None:
    """Validate BCEWithLogitsLoss inputs before computing the loss."""
    if not bool(torch_module.is_floating_point(logits)):
        raise BatchValidationError(f"{stage} logits must be floating-point tensors. Received {logits.dtype}.")
    if not bool(torch_module.is_floating_point(labels)):
        raise BatchValidationError(f"{stage} labels must be floating-point tensors. Received {labels.dtype}.")
    if tuple(logits.shape) != tuple(labels.shape):
        raise BatchValidationError(
            f"{stage} logits and labels must share the same shape. "
            f"Received {tuple(logits.shape)} and {tuple(labels.shape)}."
        )

    _ensure_finite_tensor(
        torch_module,
        logits,
        stage=stage,
        quantity="logits",
        epoch=epoch,
        batch_index=batch_index,
    )
    _ensure_finite_tensor(
        torch_module,
        labels,
        stage=stage,
        quantity="labels",
        epoch=epoch,
        batch_index=batch_index,
    )
    min_label = float(labels.detach().amin().item())
    max_label = float(labels.detach().amax().item())
    if min_label < 0.0 or max_label > 1.0:
        raise BatchValidationError(
            f"{stage} labels must stay within [0, 1] for BCEWithLogitsLoss. "
            f"Observed min={min_label:.6f}, max={max_label:.6f}."
        )


def _skippable_batch_error_fields(exc: Exception) -> tuple[str, float | None, str]:
    """Normalize skippable batch errors for logging and metrics payloads."""
    if isinstance(exc, NonFiniteValueError):
        return exc.quantity, exc.value, str(exc)
    return "value_error", None, str(exc)


def _ensure_finite_gradients(
    torch_module: Any,
    model: Any,
    *,
    epoch: int,
    batch_index: int,
) -> None:
    for parameter_name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if not bool(torch_module.isfinite(parameter.grad).all().item()):
            raise NonFiniteValueError(
                stage="train",
                quantity=f"gradient:{parameter_name}",
                epoch=epoch,
                batch_index=batch_index,
            )


def _evaluate_model(
    model: Any,
    loader: Any,
    criterion: Any,
    device: str,
    threshold: float,
    ks: list[int],
    ef_fractions: list[float],
    logger: Any | None = None,
    stage: str = "validation",
    max_consecutive_bad_batches: int = 5,
    epoch: int | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    torch = try_import_torch()
    model.eval()

    logits_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    target_ids: list[str] = []
    total_loss = 0.0
    n_batches = 0
    skipped_bad_batches = 0
    consecutive_bad_batches = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            try:
                batch = _move_batch_to_device(batch, device)
                outputs = model(batch)
                logits = outputs["logits"]
                labels = batch["labels"]
                _validate_binary_loss_inputs(
                    torch,
                    logits,
                    labels,
                    stage=stage,
                    epoch=epoch,
                    batch_index=batch_index,
                )
                loss = criterion(logits, labels)
                loss_value = _coerce_finite_float(
                    loss.item(),
                    stage=stage,
                    quantity="loss",
                    epoch=epoch,
                    batch_index=batch_index,
                )

                total_loss += loss_value
                n_batches += 1
                consecutive_bad_batches = 0
                logits_list.append(logits.detach().cpu().numpy())
                labels_list.append(labels.detach().cpu().numpy())
                target_ids.extend(batch["target_uniprot_ids"])
            except (NonFiniteValueError, BatchValidationError) as exc:
                skipped_bad_batches += 1
                consecutive_bad_batches += 1
                quantity, _, message = _skippable_batch_error_fields(exc)
                if logger is not None:
                    logger.warning(
                        "Skipping bad %s batch at epoch %s batch %s due to %s (consecutive=%s/%s): %s",
                        stage,
                        epoch,
                        batch_index,
                        quantity,
                        consecutive_bad_batches,
                        max_consecutive_bad_batches,
                        message,
                    )
                if consecutive_bad_batches >= max_consecutive_bad_batches:
                    raise NonFiniteValueError(
                        stage=stage,
                        quantity="consecutive_bad_batches",
                        value=float(consecutive_bad_batches),
                        epoch=epoch,
                        batch_index=batch_index,
                    ) from exc
                continue

    if not logits_list:
        raise NonFiniteValueError(stage=stage, quantity="all_batches_skipped", epoch=epoch)

    logits_np = np.concatenate(logits_list)
    labels_np = np.concatenate(labels_list)
    avg_loss = _coerce_finite_float(total_loss / max(n_batches, 1), stage=stage, quantity="avg_loss", epoch=epoch)
    if skipped_bad_batches > 0 and logger is not None:
        logger.warning("Skipped %s bad %s batches while computing metrics.", skipped_bad_batches, stage)
    return build_metrics_payload(
        labels=labels_np,
        logits=logits_np,
        group_ids=target_ids,
        threshold=threshold,
        ks=ks,
        ef_fractions=ef_fractions,
        loss=avg_loss,
    )


def _save_checkpoint(
    torch_module: Any,
    checkpoint_path: Path,
    model: Any,
    optimizer: Any,
    epoch_index: int,
    best_metric: float,
    config: dict[str, Any],
    val_metrics: dict[str, Any] | None = None,
    best_state: dict[str, Any] | None = None,
) -> None:
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch_index,
        "best_metric": best_metric,
        "config": config,
        "val_metrics": val_metrics,
        "best_state": best_state,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch_module.save(checkpoint, checkpoint_path)


def _load_checkpoint(torch_module: Any, checkpoint_path: Path, model: Any, optimizer: Any | None = None) -> dict[str, Any]:
    checkpoint = torch_module.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint


def _snapshot_model_state_to_cpu(torch_module: Any, model: Any) -> dict[str, Any]:
    """Clone the current model weights into host RAM for later restoration."""
    state_dict = model.state_dict()
    snapshot: dict[str, Any] = {}
    for key, value in state_dict.items():
        if torch_module.is_tensor(value):
            snapshot[key] = value.detach().cpu().clone()
        else:
            snapshot[key] = value
    return snapshot


def _write_training_metrics(
    run_dir: Path,
    config: dict[str, Any],
    status: str,
    data_summary: dict[str, Any] | None,
    best_state_payload: dict[str, Any] | None,
    history: list[dict[str, Any]],
    test_metrics: dict[str, Any] | None = None,
    training_summary: dict[str, Any] | None = None,
    termination: dict[str, Any] | None = None,
) -> None:
    payload = {
        "status": status,
        "mode": config["mode"],
        "run_name": config["run_name"],
        "environment": environment_snapshot(),
        "data": data_summary,
        "best": best_state_payload,
        "history": history,
    }
    if test_metrics is not None:
        payload["test"] = test_metrics
    if training_summary is not None:
        payload["training_summary"] = training_summary
    if termination is not None:
        payload["termination"] = termination
        if termination.get("reason") == "keyboard_interrupt":
            payload["interruption"] = termination
    write_json(round_nested(payload), run_dir / "metrics.json")


def run_training(config: dict[str, Any], dry_run: bool, max_steps: int | None = None) -> Path:
    output_cfg = config["output"]
    training_cfg = config["training"]

    run_dir, actual_run_name = prepare_run_directory(
        root_dir=output_cfg["root_dir"],
        run_name=config["run_name"],
        allow_rerun_suffix=bool(output_cfg["allow_rerun_suffix"]),
        resume=bool(training_cfg["resume"]),
    )
    config["run_name"] = actual_run_name
    config["output"]["run_dir"] = str(run_dir)
    config["output"]["checkpoint_dir"] = str(run_dir / "checkpoints")

    logger = build_logger(run_dir / "train.log", logger_name=f"train:{actual_run_name}")
    save_config(config, run_dir / "config.resolved.yaml")

    split_summary = describe_splits(config["data"])
    graph_cache_summary = describe_graph_cache(config["data"])
    data_summary: dict[str, Any] | None = round_nested(split_summary)
    logger.info("Resolved config from %s", config["config_path"])
    logger.info("Run directory: %s", run_dir)
    logger.info("Split summary: %s", round_nested(split_summary))
    logger.info("Graph cache: %s", round_nested(graph_cache_summary))

    if dry_run:
        payload = {
            "status": "dry_run",
            "mode": config["mode"],
            "run_name": config["run_name"],
            "split_summary": round_nested(split_summary),
            "graph_cache": round_nested(graph_cache_summary),
            "environment": environment_snapshot(),
        }
        write_json(payload, run_dir / "metrics.json")
        logger.info("Dry-run completed successfully.")
        return run_dir

    torch = try_import_torch()
    from src.model.dti_model import build_model

    device = choose_device(str(training_cfg["device"]), torch)
    set_global_seed(int(config["seed"]), torch_module=torch)

    model = build_model(config)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["lr"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    amp_enabled = bool(training_cfg["amp"]) and device.startswith("cuda")
    scaler = _build_grad_scaler(torch, device=device, enabled=amp_enabled)

    best_ckpt = run_dir / "checkpoints" / "best.pt"
    latest_ckpt = run_dir / "checkpoints" / "latest.pt"
    start_epoch = 0
    best_metric = float("-inf")
    best_model_state: dict[str, Any] | None = None
    best_state_payload: dict[str, Any] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    current_stage = "setup"
    current_epoch: int | None = None
    last_completed_epoch: int | None = None
    should_save_checkpoints = bool(output_cfg["save_checkpoints"])
    grad_clip_norm = training_cfg.get("grad_clip_norm")
    grad_clip_norm = None if grad_clip_norm is None else float(grad_clip_norm)
    max_consecutive_bad_batches = max(5, int(training_cfg["max_consecutive_bad_batches"]))
    skipped_bad_batches_total = 0
    skipped_bad_batch_examples: list[dict[str, Any]] = []
    max_consecutive_bad_batches_seen = 2

    if bool(training_cfg["resume"]):
        resume_checkpoint = latest_ckpt if latest_ckpt.exists() else best_ckpt if best_ckpt.exists() else None
        if resume_checkpoint is not None:
            checkpoint = _load_checkpoint(torch, resume_checkpoint, model, optimizer)
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            best_metric = float(checkpoint.get("best_metric", float("-inf")))
            checkpoint_best_state = checkpoint.get("best_state")
            if isinstance(checkpoint_best_state, dict):
                best_state_payload = checkpoint_best_state
            else:
                best_val_metrics = checkpoint.get("val_metrics")
                if isinstance(best_val_metrics, dict):
                    best_state_payload = {
                        "epoch": int(checkpoint.get("epoch", start_epoch - 1)),
                        "val": best_val_metrics,
                    }
            if best_ckpt.exists():
                best_checkpoint = torch.load(best_ckpt, map_location="cpu")
                checkpoint_model_state = best_checkpoint.get("model_state")
                if isinstance(checkpoint_model_state, dict):
                    best_model_state = checkpoint_model_state
            if best_model_state is None:
                best_model_state = _snapshot_model_state_to_cpu(torch, model)
            logger.info("Resumed from checkpoint %s at epoch %s", resume_checkpoint, start_epoch)
        else:
            logger.info("Resume requested, but no checkpoint was found under %s", run_dir / "checkpoints")

    max_train_batches = (
        int(training_cfg["max_train_batches"]) if training_cfg["max_train_batches"] is not None else max_steps
    )
    max_eval_batches = int(training_cfg["max_eval_batches"]) if training_cfg["max_eval_batches"] is not None else None
    primary_metric = str(config["metrics"]["primary"])

    def build_training_summary() -> dict[str, Any] | None:
        if skipped_bad_batches_total <= 0 and max_consecutive_bad_batches_seen <= 0:
            return None
        return {
            "skipped_bad_batches": skipped_bad_batches_total,
            "skipped_bad_batch_examples": skipped_bad_batch_examples,
            "max_consecutive_bad_batches": max_consecutive_bad_batches,
            "max_consecutive_bad_batches_seen": max_consecutive_bad_batches_seen,
        }

    try:
        current_stage = "dataloaders"
        loaders, loader_metadata = build_dataloaders(config)
        data_summary = loader_metadata["summary"]
        logger.info("Loader metadata: %s", round_nested(loader_metadata))

        current_stage = "train"
        for epoch_index in range(start_epoch, int(training_cfg["epochs"])):
            current_epoch = epoch_index
            model.train()
            epoch_loss = 0.0
            batch_count = 0
            skipped_bad_batches_epoch = 0
            consecutive_bad_batches = 0

            for batch_index, batch in enumerate(loaders["train"]):
                if max_train_batches is not None and batch_index >= max_train_batches:
                    break
                batch = _move_batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                gradients_unscaled = False

                try:
                    with _autocast_context(torch, device=device, enabled=amp_enabled):
                        outputs = model(batch)
                        logits = outputs["logits"]
                        labels = batch["labels"]

                    _validate_binary_loss_inputs(
                        torch,
                        logits,
                        labels,
                        stage="train",
                        epoch=epoch_index,
                        batch_index=batch_index,
                    )
                    with _autocast_context(torch, device=device, enabled=amp_enabled):
                        loss = criterion(logits, labels)
                    loss_value = _coerce_finite_float(
                        loss.item(),
                        stage="train",
                        quantity="loss",
                        epoch=epoch_index,
                        batch_index=batch_index,
                    )

                    scaler.scale(loss).backward()
                    if amp_enabled:
                        scaler.unscale_(optimizer)
                        gradients_unscaled = True
                    if grad_clip_norm is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                        _coerce_finite_float(
                            grad_norm,
                            stage="train",
                            quantity="grad_norm",
                            epoch=epoch_index,
                            batch_index=batch_index,
                        )
                    _ensure_finite_gradients(torch, model, epoch=epoch_index, batch_index=batch_index)
                    scaler.step(optimizer)
                    scaler.update()

                    epoch_loss += loss_value
                    batch_count += 1
                    consecutive_bad_batches *= 0.95  # Decay consecutive bad batch count to allow recovery over time
                except (NonFiniteValueError, BatchValidationError) as exc:
                    optimizer.zero_grad(set_to_none=True)
                    if amp_enabled and gradients_unscaled:
                        scaler.update()
                    skipped_bad_batches_total += 1
                    skipped_bad_batches_epoch += 1
                    consecutive_bad_batches += 1
                    max_consecutive_bad_batches_seen = max(max_consecutive_bad_batches_seen, consecutive_bad_batches)
                    quantity, observed_value, message = _skippable_batch_error_fields(exc)
                    if len(skipped_bad_batch_examples) < 10:
                        skipped_bad_batch_examples.append(
                            {
                                "epoch": epoch_index,
                                "batch_index": batch_index,
                                "quantity": quantity,
                                "observed_value": observed_value,
                                "message": message,
                            }
                        )
                    logger.warning(
                        "Skipping bad training batch at epoch %s batch %s due to %s (consecutive=%s/%s): %s",
                        epoch_index,
                        batch_index,
                        quantity,
                        consecutive_bad_batches,
                        max_consecutive_bad_batches,
                        message,
                    )
                    if consecutive_bad_batches >= max_consecutive_bad_batches:
                        raise NonFiniteValueError(
                            stage="train",
                            quantity="consecutive_bad_batches",
                            value=float(consecutive_bad_batches),
                            epoch=epoch_index,
                            batch_index=batch_index,
                        ) from exc
                    continue

            if batch_count == 0:
                raise NonFiniteValueError(stage="train", quantity="all_batches_skipped", epoch=epoch_index)

            train_loss = _coerce_finite_float(epoch_loss / batch_count, stage="train", quantity="avg_loss", epoch=epoch_index)
            current_stage = "validation"
            val_metrics = _evaluate_model(
                model=model,
                loader=loaders["val"],
                criterion=criterion,
                device=device,
                threshold=float(config["metrics"]["threshold"]),
                ks=[int(item) for item in config["metrics"]["ks"]],
                ef_fractions=[float(item) for item in config["metrics"]["ef_fractions"]],
                logger=logger,
                stage="validation",
                max_consecutive_bad_batches=max_consecutive_bad_batches,
                epoch=epoch_index,
                max_batches=max_eval_batches,
            )

            metric_value = val_metrics["classification"].get(primary_metric)
            if metric_value is not None:
                _coerce_finite_float(metric_value, stage="validation", quantity=f"metric:{primary_metric}", epoch=epoch_index)
            metric_score = float("-inf") if metric_value is None else float(metric_value)
            history.append(
                {
                    "epoch": epoch_index,
                    "train_loss": train_loss,
                    "train_batches": batch_count,
                    "skipped_bad_batches": skipped_bad_batches_epoch,
                    "val": val_metrics,
                }
            )
            logger.info(
                "Epoch %s | train_loss=%.4f | skipped_bad_batches=%s | val_%s=%s",
                epoch_index,
                train_loss,
                skipped_bad_batches_epoch,
                primary_metric,
                "nan" if metric_value is None else f"{metric_value:.4f}",
            )

            should_update_best = metric_value is not None and (metric_score > best_metric or not best_ckpt.exists())
            if should_update_best:
                best_metric = metric_score
                epochs_without_improvement = 0
                best_model_state = _snapshot_model_state_to_cpu(torch, model)
                best_state_payload = {"epoch": epoch_index, "val": val_metrics}
                if should_save_checkpoints:
                    _save_checkpoint(
                        torch_module=torch,
                        checkpoint_path=best_ckpt,
                        model=model,
                        optimizer=optimizer,
                        epoch_index=epoch_index,
                        best_metric=best_metric,
                        config=config,
                        val_metrics=val_metrics,
                        best_state=best_state_payload,
                    )
            else:
                epochs_without_improvement += 1

            if should_save_checkpoints:
                _save_checkpoint(
                    torch_module=torch,
                    checkpoint_path=latest_ckpt,
                    model=model,
                    optimizer=optimizer,
                    epoch_index=epoch_index,
                    best_metric=best_metric,
                    config=config,
                    val_metrics=val_metrics,
                    best_state=best_state_payload,
                )

            last_completed_epoch = epoch_index
            current_stage = "train"

            if epochs_without_improvement >= int(training_cfg["early_stopping_patience"]):
                logger.info("Early stopping triggered after %s epochs without improvement.", epochs_without_improvement)
                break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        elif best_ckpt.exists():
            _load_checkpoint(torch, best_ckpt, model)

        current_stage = "test"
        test_metrics = _evaluate_model(
            model=model,
            loader=loaders["test"],
            criterion=criterion,
            device=device,
            threshold=float(config["metrics"]["threshold"]),
            ks=[int(item) for item in config["metrics"]["ks"]],
            ef_fractions=[float(item) for item in config["metrics"]["ef_fractions"]],
            logger=logger,
            stage="test",
            max_consecutive_bad_batches=max_consecutive_bad_batches,
            max_batches=max_eval_batches,
        )

        _write_training_metrics(
            run_dir=run_dir,
            config=config,
            status="completed",
            data_summary=data_summary,
            best_state_payload=best_state_payload,
            history=history,
            training_summary=build_training_summary(),
            test_metrics=test_metrics,
        )
        from src.eval import write_evaluation_artifacts

        evaluation_source = (
            "post_train_in_memory_best"
            if best_model_state is not None
            else "post_train_checkpoint"
            if best_ckpt.exists()
            else "post_train_current_model"
        )
        write_evaluation_artifacts(
            config=config,
            data_summary=data_summary,
            test_metrics=test_metrics,
            checkpoint_path=str(best_ckpt) if best_ckpt.exists() else None,
            source=evaluation_source,
        )
        logger.info(
            "Training finished. Metrics written to %s and %s",
            run_dir / "metrics.json",
            run_dir / "eval_metrics.json",
        )
        return run_dir
    except NonFiniteValueError as exc:
        resume_checkpoint = latest_ckpt if latest_ckpt.exists() else best_ckpt if best_ckpt.exists() else None
        best_checkpoint = best_ckpt if best_ckpt.exists() else None
        termination_payload = exc.to_payload(
            resume_checkpoint=resume_checkpoint,
            best_checkpoint=best_checkpoint,
            best_metric=best_metric if math.isfinite(best_metric) else None,
        )
        logger.error("%s", exc)

        test_metrics: dict[str, Any] | None = None
        status = "failed_non_finite"
        if best_model_state is not None or best_checkpoint is not None:
            if best_model_state is not None:
                logger.warning("Reloading best in-memory model state after non-finite stop.")
                model.load_state_dict(best_model_state)
            elif best_checkpoint is not None:
                logger.warning("Reloading best checkpoint from %s after non-finite stop.", best_checkpoint)
                _load_checkpoint(torch, best_checkpoint, model)
            current_stage = "test"
            test_metrics = _evaluate_model(
                model=model,
                loader=loaders["test"],
                criterion=criterion,
                device=device,
                threshold=float(config["metrics"]["threshold"]),
                ks=[int(item) for item in config["metrics"]["ks"]],
                ef_fractions=[float(item) for item in config["metrics"]["ef_fractions"]],
                logger=logger,
                stage="test",
                max_consecutive_bad_batches=max_consecutive_bad_batches,
                max_batches=max_eval_batches,
            )
            status = "completed_with_non_finite_stop"

        _write_training_metrics(
            run_dir=run_dir,
            config=config,
            status=status,
            data_summary=data_summary,
            best_state_payload=best_state_payload,
            history=history,
            training_summary=build_training_summary(),
            test_metrics=test_metrics,
            termination=termination_payload,
        )
        if test_metrics is not None:
            from src.eval import write_evaluation_artifacts

            evaluation_source = (
                "post_train_non_finite_in_memory_best"
                if best_model_state is not None
                else "post_train_non_finite_checkpoint"
                if best_checkpoint is not None
                else "post_train_non_finite_current_model"
            )
            write_evaluation_artifacts(
                config=config,
                data_summary=data_summary,
                test_metrics=test_metrics,
                checkpoint_path=str(best_checkpoint) if best_checkpoint is not None else None,
                source=evaluation_source,
                status="completed",
            )
        logger.warning(
            "Training stopped due to non-finite values during %s. Metrics written to %s",
            exc.stage,
            run_dir / "metrics.json",
        )
        return run_dir
    except KeyboardInterrupt:
        resume_checkpoint = latest_ckpt if latest_ckpt.exists() else best_ckpt if best_ckpt.exists() else None
        termination_payload = {
            "reason": "keyboard_interrupt",
            "stage": current_stage,
            "epoch": current_epoch,
            "last_completed_epoch": last_completed_epoch,
            "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
        }
        _write_training_metrics(
            run_dir=run_dir,
            config=config,
            status="interrupted",
            data_summary=data_summary,
            best_state_payload=best_state_payload,
            history=history,
            training_summary=build_training_summary(),
            termination=termination_payload,
        )
        logger.warning(
            "Training interrupted by user during %s. Partial metrics written to %s",
            current_stage,
            run_dir / "metrics.json",
        )
        if resume_checkpoint is not None:
            logger.warning("Resume is available from checkpoint %s", resume_checkpoint)
        raise


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a cold-protein DTI model from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to the experiment config file.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    parser.add_argument("--set", action="append", default=[], help="Override a config value with key=value.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and outputs without importing torch.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional training batch limit for smoke tests.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        config = load_and_resolve_config(
            config_path=args.config,
            seed=args.seed,
            mode="train",
            overrides=args.set,
        )
        run_training(config=config, dry_run=bool(args.dry_run), max_steps=args.max_steps)
    except ConfigError as exc:
        raise SystemExit(f"[config-error] {exc}") from exc
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
