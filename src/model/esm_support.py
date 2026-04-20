"""Shared helpers for loading staged local protein language model backbones."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LoadedESMBackbone:
    """Loaded protein LM backbone plus the tokenizer needed to drive it."""

    backend: str
    backbone: Any
    tokenizer: Any
    num_layers: int
    hidden_size: int
    resolved_path: str | None = None


ESMC_MODEL_SPECS: dict[str, dict[str, Any]] = {
    "esmc_600m": {
        "aliases": {
            "esmc_600m",
            "esmc-600m",
            "esmc-600m-2024-12",
            "evolutionaryscale/esmc-600m-2024-12",
        },
        "d_model": 1152,
        "n_heads": 18,
        "n_layers": 36,
        "weight_path": Path("data") / "weights" / "esmc_600m_2024_12_v0.pth",
    },
}

VESM_BASE_MODELS = {
    "VESM_650M": "esm2_t33_650M_UR50D",
}


def _canonical_model_name(model_name: str) -> str:
    lowered = model_name.replace("\\", "/").lower()
    for canonical, spec in ESMC_MODEL_SPECS.items():
        if lowered in spec["aliases"]:
            return canonical
    if lowered in {"vesm_650m", "vesm-650m"}:
        return "VESM_650M"
    return model_name


def _artifact_names(model_name: str) -> list[str]:
    canonical_name = _canonical_model_name(model_name)
    names = [
        model_name,
        canonical_name,
        model_name.replace("\\", "/").split("/")[-1],
        model_name.replace("\\", "/").replace("/", "__"),
    ]
    deduped: list[str] = []
    for name in names:
        if name and name not in deduped:
            deduped.append(name)
    return deduped


def discover_local_esm_artifact(model_name: str, search_root: str | Path = "artifacts/pretrained") -> Path | None:
    """Return the first staged local artifact matching the requested model name."""

    root = Path(search_root)
    candidates = [root / name for name in _artifact_names(model_name)]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _require_nonempty_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if not path.is_file():
        raise ValueError(f"Expected {label} to be a file: {path}")
    if path.stat().st_size == 0:
        raise RuntimeError(f"{label} appears incomplete because it is zero bytes: {path}")


def _resolve_weight_file(model_path: Path, relative_paths: list[Path]) -> Path:
    if model_path.is_file():
        _require_nonempty_file(model_path, "model weights")
        return model_path

    for relative_path in relative_paths:
        candidate = model_path / relative_path
        if candidate.exists():
            _require_nonempty_file(candidate, "model weights")
            return candidate

    expected = ", ".join(str(path) for path in relative_paths)
    raise FileNotFoundError(f"Could not find model weights under {model_path}. Expected one of: {expected}")


def _torch_load_state_dict(path: Path) -> dict[str, Any]:
    import torch

    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        loaded = torch.load(path, map_location="cpu")

    if not isinstance(loaded, dict):
        raise RuntimeError(f"Expected a state dict in {path}, found {type(loaded).__name__}.")

    for key in ("state_dict", "model_state_dict", "model"):
        nested = loaded.get(key)
        if isinstance(nested, dict):
            return nested
    return loaded


def _load_hf_esm_backbone(model_path: Path, overlay_state_path: Path | None = None) -> LoadedESMBackbone:
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
        if path.suffix not in {".json"}:
            _require_nonempty_file(path, "local ESM model weights")

    try:
        from transformers import AutoModelForMaskedLM, AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Loading a staged Hugging Face ESM model requires the optional `transformers` dependency. "
            "Install it with `uv sync --extra esm`."
        ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
        masked_lm_model = AutoModelForMaskedLM.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=False,
        )
    except OSError as exc:
        raise RuntimeError(
            f"Unable to load the staged ESM model from {model_path}. "
            "Check that the tokenizer files and weights are fully downloaded."
        ) from exc

    if overlay_state_path is not None:
        _require_nonempty_file(overlay_state_path, "overlay model weights")
        state_dict = _torch_load_state_dict(overlay_state_path)
        model_keys = set(masked_lm_model.state_dict())
        matched_keys = model_keys.intersection(state_dict)
        if not matched_keys:
            raise RuntimeError(
                f"Overlay weights in {overlay_state_path} did not match any parameters in the base ESM model."
            )
        masked_lm_model.load_state_dict(state_dict, strict=False)

    backbone = getattr(masked_lm_model, "esm", None)
    if backbone is None:
        raise RuntimeError(
            f"Loaded ESM checkpoint from {model_path}, but could not locate the base encoder on the masked-LM wrapper."
        )

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


def _load_vesm_backbone(
    model_name: str,
    model_path: Path,
    base_model_name: str | None,
    base_checkpoint_path: str | None,
    prefer_staged_artifacts: bool,
) -> LoadedESMBackbone:
    canonical_name = _canonical_model_name(model_name)
    if canonical_name not in VESM_BASE_MODELS:
        available = ", ".join(sorted(VESM_BASE_MODELS))
        raise ValueError(f"Unsupported VESM model '{model_name}'. Available choices: {available}")

    overlay_path = _resolve_weight_file(model_path, [Path(f"{canonical_name}.pth")])
    resolved_base_model = base_model_name or VESM_BASE_MODELS[canonical_name]
    if base_checkpoint_path:
        base_path = Path(base_checkpoint_path)
        if not base_path.exists():
            raise FileNotFoundError(f"Local VESM base checkpoint not found: {base_path}")
    elif prefer_staged_artifacts:
        base_path = discover_local_esm_artifact(resolved_base_model)  # type: ignore[assignment]
        if base_path is None:
            raise FileNotFoundError(
                "Could not find the staged ESM2 base model needed for VESM. "
                f"Expected `artifacts/pretrained/{resolved_base_model}` or pass "
                "`model.protein_encoder.base_checkpoint_path` explicitly."
            )
    else:
        raise FileNotFoundError(
            "VESM requires an ESM2 base checkpoint. Pass "
            "`model.protein_encoder.base_checkpoint_path` or enable "
            "`model.protein_encoder.prefer_staged_artifacts`."
        )

    if not base_path.is_dir():
        raise ValueError(f"VESM base checkpoints must be staged as a model directory: {base_path}")

    loaded = _load_hf_esm_backbone(base_path, overlay_state_path=overlay_path)
    loaded.resolved_path = f"{overlay_path} (base: {base_path})"
    return loaded


def _load_esmc_backbone(model_name: str, model_path: Path) -> LoadedESMBackbone:
    canonical_name = _canonical_model_name(model_name)
    spec = ESMC_MODEL_SPECS.get(canonical_name)
    if spec is None:
        available = ", ".join(sorted(ESMC_MODEL_SPECS))
        raise ValueError(f"Unsupported ESMC model '{model_name}'. Available choices: {available}")

    weight_path = _resolve_weight_file(model_path, [spec["weight_path"], Path(spec["weight_path"]).name])

    try:
        from esm.models.esmc import ESMC  # type: ignore
        from esm.tokenization import get_esmc_model_tokenizers  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Loading ESM C requires the optional EvolutionaryScale `esm` package. "
            "Install it with `uv sync --extra esmc`. This project pins that extra to "
            "the Python 3.10-compatible 3.1.x release line."
        ) from exc

    model = ESMC(
        d_model=int(spec["d_model"]),
        n_heads=int(spec["n_heads"]),
        n_layers=int(spec["n_layers"]),
        tokenizer=get_esmc_model_tokenizers(),
        use_flash_attn=False,
    ).eval()
    model.load_state_dict(_torch_load_state_dict(weight_path))

    return LoadedESMBackbone(
        backend="esmc",
        backbone=model,
        tokenizer=model.tokenizer,
        num_layers=int(spec["n_layers"]),
        hidden_size=int(spec["d_model"]),
        resolved_path=str(model_path),
    )


def _infer_backend(model_name: str, backend: str | None) -> str:
    if backend:
        return backend.lower().replace("-", "_")

    canonical_name = _canonical_model_name(model_name).lower()
    if canonical_name.startswith("esmc"):
        return "esmc"
    if canonical_name.startswith("vesm"):
        return "vesm"
    return "huggingface"


def _resolve_checkpoint_path(
    model_name: str,
    local_checkpoint_path: str | None,
    prefer_staged_artifacts: bool,
) -> Path | None:
    if local_checkpoint_path:
        checkpoint_path = Path(local_checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Local protein LM checkpoint not found: {checkpoint_path}")
        return checkpoint_path
    if prefer_staged_artifacts:
        return discover_local_esm_artifact(model_name)
    return None


def load_esm_backbone(
    model_name: str,
    local_checkpoint_path: str | None = None,
    prefer_staged_artifacts: bool = True,
    backend: str | None = None,
    base_model_name: str | None = None,
    base_checkpoint_path: str | None = None,
) -> LoadedESMBackbone:
    """Load a protein LM backbone, preferring staged local artifacts and never downloading implicitly."""

    resolved_backend = _infer_backend(model_name, backend)
    checkpoint_path = _resolve_checkpoint_path(model_name, local_checkpoint_path, prefer_staged_artifacts)
    if checkpoint_path is None:
        raise FileNotFoundError(
            "Could not find a staged local protein LM artifact. "
            f"Expected `artifacts/pretrained/{model_name}` or pass "
            "`model.protein_encoder.local_checkpoint_path` explicitly. "
            "No remote download is attempted during model construction."
        )

    if resolved_backend == "vesm":
        return _load_vesm_backbone(
            model_name=model_name,
            model_path=checkpoint_path,
            base_model_name=base_model_name,
            base_checkpoint_path=base_checkpoint_path,
            prefer_staged_artifacts=prefer_staged_artifacts,
        )

    if resolved_backend == "esmc":
        return _load_esmc_backbone(model_name=model_name, model_path=checkpoint_path)

    if resolved_backend not in {"huggingface", "hf", "esm"}:
        raise ValueError(f"Unsupported protein LM backend '{resolved_backend}'.")

    if not checkpoint_path.is_dir():
        raise ValueError(
            f"ESM checkpoints must be staged as a model directory, not a file: {checkpoint_path}"
        )
    return _load_hf_esm_backbone(checkpoint_path)
