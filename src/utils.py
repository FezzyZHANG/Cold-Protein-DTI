"""Utility helpers for experiment setup, logging, and file outputs."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
import logging
from pathlib import Path
import platform
import random
import sys
from typing import Any

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def prepare_run_directory(
    root_dir: str | Path,
    run_name: str,
    allow_rerun_suffix: bool = True,
    resume: bool = False,
) -> tuple[Path, str]:
    root = ensure_dir(root_dir)
    run_dir = root / run_name

    if resume:
        ensure_dir(run_dir / "checkpoints")
        return run_dir, run_name

    if run_dir.exists() and any(run_dir.iterdir()):
        if not allow_rerun_suffix:
            raise FileExistsError(
                f"Run directory already exists and is not empty: {run_dir}. "
                "Enable `output.allow_rerun_suffix` or set `training.resume`."
            )
        run_name = f"{run_name}__{timestamp_slug()}"
        run_dir = root / run_name

    ensure_dir(run_dir / "checkpoints")
    return run_dir, run_name


def build_logger(log_path: str | Path, logger_name: str) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def environment_snapshot() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def set_global_seed(seed: int, torch_module: Any | None = None) -> None:
    """Set Python, NumPy, and optional torch RNG state."""
    random.seed(seed)
    np.random.seed(seed)

    if torch_module is None:
        return

    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)

    try:
        torch_module.backends.cudnn.deterministic = True
        torch_module.backends.cudnn.benchmark = False
    except AttributeError:
        pass


def choose_device(requested: str, torch_module: Any) -> str:
    if requested == "cuda" and torch_module.cuda.is_available():
        return "cuda"
    if requested.startswith("cuda") and torch_module.cuda.is_available():
        return requested
    return "cpu"


def try_import_torch() -> Any:
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for train/eval execution. Install the training extra, "
            "for example with `uv sync --extra train`."
        ) from exc
    return torch


def round_nested(value: Any, digits: int = 6) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {key: round_nested(item, digits=digits) for key, item in value.items()}
    if isinstance(value, list):
        return [round_nested(item, digits=digits) for item in value]
    return value

