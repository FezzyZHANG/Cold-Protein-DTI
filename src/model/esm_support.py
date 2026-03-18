"""Shared helpers for loading staged local ESM backbones."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LoadedESMBackbone:
    """Loaded ESM backbone plus the tokenizer needed to drive it."""

    backend: str
    backbone: Any
    tokenizer: Any
    num_layers: int
    hidden_size: int
    resolved_path: str | None = None


def discover_local_esm_artifact(model_name: str, search_root: str | Path = "artifacts/pretrained") -> Path | None:
    """Return the first staged local artifact matching the requested model name."""

    root = Path(search_root)
    candidates = [root / model_name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_hf_esm_backbone(model_path: Path) -> LoadedESMBackbone:
    required_files = [
        model_path / "config.json",
        model_path / "tokenizer_config.json",
        model_path / "vocab.txt",
    ]
    missing_files = [path.name for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Local ESM model directory is missing required files: {', '.join(sorted(missing_files))} ({model_path})"
        )

    weight_files = [
        model_path / "model.safetensors",
        model_path / "pytorch_model.bin",
        model_path / "model.safetensors.index.json",
        model_path / "pytorch_model.bin.index.json",
    ]
    present_weights = [path for path in weight_files if path.exists()]
    if not present_weights:
        raise FileNotFoundError(
            "Local ESM model directory does not contain model weights. "
            f"Expected one of: {', '.join(path.name for path in weight_files)}"
        )

    for path in present_weights:
        if path.suffix in {".json"}:
            continue
        if path.stat().st_size == 0:
            raise RuntimeError(
                "Local ESM model weights appear incomplete. "
                f"Found zero-byte file: {path}. Re-download the model before training."
            )

    try:
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Loading a staged Hugging Face ESM model requires the optional `transformers` dependency. "
            "Install it with `uv sync --extra esm`."
        ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
        backbone = AutoModel.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    except OSError as exc:
        raise RuntimeError(
            f"Unable to load the staged ESM model from {model_path}. "
            "Check that the tokenizer files and weights are fully downloaded."
        ) from exc

    num_layers = getattr(backbone.config, "num_hidden_layers", None)
    hidden_size = getattr(backbone.config, "hidden_size", None)
    if num_layers is None or hidden_size is None:
        raise RuntimeError(f"Unable to infer ESM backbone metadata from {model_path}.")

    return LoadedESMBackbone(
        backend="huggingface",
        backbone=backbone,
        tokenizer=tokenizer,
        num_layers=int(num_layers),
        hidden_size=int(hidden_size),
        resolved_path=str(model_path),
    )


def load_esm_backbone(
    model_name: str,
    local_checkpoint_path: str | None = None,
    prefer_staged_artifacts: bool = True,
) -> LoadedESMBackbone:
    """Load an ESM backbone from a staged local model directory."""

    checkpoint_path: Path | None = None
    if local_checkpoint_path:
        checkpoint_path = Path(local_checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Local ESM checkpoint not found: {checkpoint_path}")
    elif prefer_staged_artifacts:
        checkpoint_path = discover_local_esm_artifact(model_name)

    if checkpoint_path is None:
        raise FileNotFoundError(
            "Could not find a staged local ESM model directory. "
            f"Expected `artifacts/pretrained/{model_name}` or pass "
            "`model.protein_encoder.local_checkpoint_path` explicitly."
        )
    if not checkpoint_path.is_dir():
        raise ValueError(
            f"ESM checkpoints must be staged as a model directory, not a file: {checkpoint_path}"
        )
    return _load_hf_esm_backbone(checkpoint_path)
