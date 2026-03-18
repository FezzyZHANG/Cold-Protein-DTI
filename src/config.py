"""Configuration loading, normalization, and persistence helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from typing import Any


class ConfigError(ValueError):
    """Raised when a config file is missing required fields or uses invalid values."""


_FLOAT_RE = re.compile(r"^[+-]?(?:\d+\.\d*|\d+|\.\d+)(?:[eE][+-]?\d+)?$")
_INT_RE = re.compile(r"^[+-]?\d+$")


def _split_inline_list(text: str) -> list[str]:
    items: list[str] = []
    current: list[str] = []
    quote: str | None = None

    for char in text:
        if quote:
            current.append(char)
            if char == quote:
                quote = None
            continue

        if char in {"'", '"'}:
            quote = char
            current.append(char)
            continue

        if char == ",":
            items.append("".join(current).strip())
            current = []
            continue

        current.append(char)

    if current:
        items.append("".join(current).strip())
    return [item for item in items if item]


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""

    lowered = value.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item) for item in _split_inline_list(inner)]

    if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
        return value[1:-1]

    if _INT_RE.match(value):
        return int(value)
    if _FLOAT_RE.match(value):
        return float(value)
    return value


def _strip_comment(line: str) -> str:
    quote: str | None = None
    result: list[str] = []

    for char in line:
        if quote:
            result.append(char)
            if char == quote:
                quote = None
            continue

        if char in {"'", '"'}:
            quote = char
            result.append(char)
            continue

        if char == "#":
            break

        result.append(char)

    return "".join(result).rstrip()


def _parse_key_value(content: str) -> tuple[str, str]:
    if ":" not in content:
        raise ConfigError(f"Expected a key/value pair, found: {content!r}")
    key, value = content.split(":", 1)
    key = key.strip()
    if not key:
        raise ConfigError(f"Encountered an empty key in line: {content!r}")
    return key, value.strip()


def _collect_yaml_lines(text: str) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        stripped_line = _strip_comment(raw_line)
        if not stripped_line.strip():
            continue
        if "\t" in raw_line:
            raise ConfigError("Tabs are not supported in the fallback YAML parser.")
        indent = len(stripped_line) - len(stripped_line.lstrip(" "))
        lines.append((indent, stripped_line.strip()))
    return lines


def _parse_yaml_block(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[Any, int]:
    if index >= len(lines):
        return {}, index

    container: dict[str, Any] | list[Any] | None = None

    while index < len(lines):
        current_indent, content = lines[index]
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ConfigError(f"Unexpected indentation near: {content!r}")

        if content.startswith("- "):
            if container is None:
                container = []
            if not isinstance(container, list):
                raise ConfigError("Cannot mix mapping and list items at the same indentation level.")

            value_part = content[2:].strip()
            if not value_part:
                item, index = _parse_yaml_block(lines, index + 1, indent + 2)
                container.append(item)
                continue

            container.append(_parse_scalar(value_part))
            index += 1
            continue

        if container is None:
            container = {}
        if not isinstance(container, dict):
            raise ConfigError("Cannot mix list and mapping items at the same indentation level.")

        key, value_part = _parse_key_value(content)
        if value_part:
            container[key] = _parse_scalar(value_part)
            index += 1
            continue

        next_index = index + 1
        if next_index < len(lines) and lines[next_index][0] > indent:
            nested_value, index = _parse_yaml_block(lines, next_index, lines[next_index][0])
            container[key] = nested_value
        else:
            container[key] = None
            index += 1

    if container is None:
        return {}, index
    return container, index


def _simple_yaml_load(text: str) -> dict[str, Any]:
    lines = _collect_yaml_lines(text)
    if not lines:
        return {}
    value, _ = _parse_yaml_block(lines, 0, lines[0][0])
    if not isinstance(value, dict):
        raise ConfigError("Top-level YAML content must be a mapping.")
    return value


def _format_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "" or text.strip() != text or ":" in text or "#" in text:
        return f'"{text}"'
    return text


def _simple_yaml_dump(value: Any, indent: int = 0) -> str:
    prefix = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(_simple_yaml_dump(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_format_scalar(item)}")
        return "\n".join(lines)

    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.append(_simple_yaml_dump(item, indent + 2))
            else:
                lines.append(f"{prefix}- {_format_scalar(item)}")
        return "\n".join(lines)

    return f"{prefix}{_format_scalar(value)}"


def _yaml_load(text: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        return _simple_yaml_load(text)

    loaded = yaml.safe_load(text)
    return {} if loaded is None else loaded


def _yaml_dump(data: dict[str, Any]) -> str:
    try:
        import yaml  # type: ignore
    except ImportError:
        return _simple_yaml_dump(data) + "\n"

    return yaml.safe_dump(data, sort_keys=False, allow_unicode=False)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file does not exist: {config_path}")
    return _yaml_load(config_path.read_text(encoding="utf-8"))


def save_config(config: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_yaml_dump(config), encoding="utf-8")


def _coerce_override_value(raw_value: str) -> Any:
    return _parse_scalar(raw_value)


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    updated = deepcopy(config)

    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"Overrides must use key=value form. Received: {item!r}")

        dotted_key, raw_value = item.split("=", 1)
        cursor: Any = updated
        parts = [part.strip() for part in dotted_key.split(".") if part.strip()]
        if not parts:
            raise ConfigError(f"Invalid override key: {item!r}")

        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]

        cursor[parts[-1]] = _coerce_override_value(raw_value)

    return updated


def _normalize_model_config(model_cfg: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(model_cfg)

    drug_cfg = normalized.get("drug_encoder", "gvp")
    if isinstance(drug_cfg, str):
        drug_cfg = {"name": drug_cfg}
    drug_cfg.setdefault("name", "gvp")
    drug_cfg.setdefault("node_hidden_scalar", normalized.get("hidden_dim", 256))
    drug_cfg.setdefault("node_hidden_vector", 16)
    drug_cfg.setdefault("edge_hidden_scalar", 64)
    drug_cfg.setdefault("edge_hidden_vector", 4)
    drug_cfg.setdefault("num_layers", 3)
    drug_cfg.setdefault("dropout", normalized.get("dropout", 0.1))
    if drug_cfg["name"] != "gvp":
        raise ConfigError("Only the conservative `gvp` drug encoder is currently supported.")

    protein_cfg = normalized.get("protein_encoder", "cnn")
    if isinstance(protein_cfg, str):
        protein_cfg = {"name": protein_cfg}
    protein_cfg["name"] = str(protein_cfg.get("name", "cnn")).lower()
    if protein_cfg["name"] == "plm":
        protein_cfg["name"] = "esm"

    protein_mode = protein_cfg.get("mode") or normalized.get("protein_encoder_mode")
    if protein_mode is None:
        protein_mode = "scratch" if protein_cfg["name"] == "cnn" else "frozen"
    protein_cfg["mode"] = str(protein_mode).lower()
    protein_cfg.setdefault("embed_dim", 128)
    protein_cfg.setdefault("hidden_dim", 256)
    protein_cfg.setdefault("kernel_sizes", [5, 9, 15])
    protein_cfg.setdefault("dropout", normalized.get("dropout", 0.1))
    protein_cfg.setdefault("model_name", "esm2_t33_650M_UR50D")
    protein_cfg.setdefault("local_checkpoint_path", None)
    protein_cfg.setdefault("repr_layer", None)
    protein_cfg.setdefault("max_input_length", 1024)

    if protein_cfg["name"] not in {"cnn", "esm"}:
        raise ConfigError("Protein encoder must be one of: cnn, esm.")

    if protein_cfg["name"] == "cnn" and protein_cfg["mode"] != "scratch":
        raise ConfigError("The CNN protein encoder only supports `mode: scratch`.")

    if protein_cfg["name"] == "esm" and protein_cfg["mode"] not in {"frozen", "finetuned"}:
        raise ConfigError("The ESM protein encoder supports `mode: frozen` or `mode: finetuned`.")

    fusion_cfg = normalized.get("fusion", "concat")
    if isinstance(fusion_cfg, str):
        fusion_cfg = {"name": fusion_cfg}
    fusion_cfg["name"] = str(fusion_cfg.get("name", "concat")).lower()
    fusion_cfg.setdefault("hidden_dim", normalized.get("hidden_dim", 256))
    fusion_cfg.setdefault("glimpses", 2)
    fusion_cfg.setdefault("dropout", normalized.get("dropout", 0.1))
    if fusion_cfg["name"] not in {"concat", "ban"}:
        raise ConfigError("Fusion head must be one of: concat, ban.")

    normalized["dropout"] = float(normalized.get("dropout", 0.1))
    normalized["drug_encoder"] = drug_cfg
    normalized["protein_encoder"] = protein_cfg
    normalized["fusion"] = fusion_cfg
    normalized["hidden_dim"] = int(normalized.get("hidden_dim", 256))
    return normalized


def _normalize_data_config(data_cfg: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(data_cfg)
    normalized.setdefault("dataset_name", "scope_dti")
    normalized.setdefault("split_name", "cp_easy")
    normalized.setdefault("max_protein_length", 1024)
    normalized.setdefault("file_format", "parquet")
    normalized.setdefault("raw_path", None)
    normalized.setdefault("graph_id_column", "inchi_key")

    split_dir = normalized.get("split_dir")
    if split_dir:
        split_dir_path = Path(split_dir)
        normalized.setdefault("train_path", str(split_dir_path / "train.parquet"))
        normalized.setdefault("val_path", str(split_dir_path / "val.parquet"))
        normalized.setdefault("test_path", str(split_dir_path / "test.parquet"))

    if not normalized.get("graph_cache_path") and normalized.get("raw_path"):
        raw_path = Path(normalized["raw_path"])
        normalized["graph_cache_path"] = str(raw_path.parent / "graphs" / f"{raw_path.stem}_graphs.pt")

    required_paths = ("train_path", "val_path", "test_path")
    if not all(normalized.get(key) for key in required_paths):
        raise ConfigError(
            "Data config must define train/val/test paths directly or via `data.split_dir`."
        )

    return normalized


def _normalize_training_config(training_cfg: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(training_cfg)
    normalized.setdefault("epochs", 30)
    normalized.setdefault("batch_size", 64)
    normalized.setdefault("lr", 1.0e-3)
    normalized.setdefault("weight_decay", 1.0e-5)
    normalized.setdefault("amp", True)
    normalized.setdefault("num_workers", 0)
    normalized.setdefault("device", "cuda")
    normalized.setdefault("early_stopping_patience", 5)
    normalized.setdefault("resume", False)
    normalized.setdefault("max_train_batches", None)
    normalized.setdefault("max_eval_batches", None)
    return normalized


def _normalize_output_config(output_cfg: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(output_cfg)
    normalized.setdefault("root_dir", "results")
    normalized.setdefault("save_checkpoints", True)
    normalized.setdefault("save_every_n_epochs", 1)
    normalized.setdefault("allow_rerun_suffix", True)
    return normalized


def _normalize_metrics_config(metrics_cfg: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(metrics_cfg)
    normalized.setdefault("primary", "auprc")
    normalized.setdefault("threshold", 0.5)
    normalized.setdefault("extra", ["auroc", "f1", "mcc"])
    normalized.setdefault("ks", [10, 50, 100])
    normalized.setdefault("ef_fractions", [0.01, 0.05])
    return normalized


def build_default_run_name(config: dict[str, Any]) -> str:
    split_name = str(config["data"]["split_name"])
    protein_cfg = config["model"]["protein_encoder"]
    fusion_name = str(config["model"]["fusion"]["name"])
    protein_descriptor = protein_cfg["name"]
    if protein_cfg["name"] != "cnn":
        protein_descriptor = f"{protein_cfg['name']}_{protein_cfg['mode']}"
    return f"{split_name}_{protein_descriptor}_{fusion_name}_s{config['seed']}"


def resolve_config(
    config: dict[str, Any],
    config_path: str | Path,
    cli_mode: str | None = None,
    cli_seed: int | None = None,
) -> dict[str, Any]:
    raw = deepcopy(config)

    if "data" not in raw or "model" not in raw or "training" not in raw:
        raise ConfigError("Config must define `data`, `model`, and `training` sections.")

    raw["mode"] = str(cli_mode or raw.get("mode", "train")).lower()
    raw["seed"] = int(cli_seed if cli_seed is not None else raw.get("seed", 42))
    raw["data"] = _normalize_data_config(raw["data"])
    raw["model"] = _normalize_model_config(raw["model"])
    raw["training"] = _normalize_training_config(raw["training"])
    raw["output"] = _normalize_output_config(raw.get("output", {}))
    raw["metrics"] = _normalize_metrics_config(raw.get("metrics", {}))
    raw["config_path"] = str(Path(config_path))
    raw["run_name"] = str(raw.get("run_name") or build_default_run_name(raw))
    raw.setdefault("artifacts", {})
    return raw


def load_and_resolve_config(
    config_path: str | Path,
    seed: int | None = None,
    mode: str | None = None,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    loaded = load_config(config_path)
    if overrides:
        loaded = apply_overrides(loaded, overrides)
    return resolve_config(loaded, config_path=config_path, cli_mode=mode, cli_seed=seed)
