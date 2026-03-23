#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"
MODE="${MODE:-cp-easy}"
SUBSAMPLE_N="${SUBSAMPLE_N:-500000}"
INPUT_PATH="${INPUT_PATH:-$PROJECT_ROOT/data/scope_dti_with_inchikey.parquet}"
SKIP_EVAL="${SKIP_EVAL:-0}"
ACTIVE_CHILD_PID=""

handle_interrupt() {
  echo "[pretest] interrupt received. Stopping active command..." >&2
  if [[ -n "${ACTIVE_CHILD_PID:-}" ]] && kill -0 "$ACTIVE_CHILD_PID" >/dev/null 2>&1; then
    kill -INT "$ACTIVE_CHILD_PID" >/dev/null 2>&1 || true
    set +e
    wait "$ACTIVE_CHILD_PID"
    set -e
    ACTIVE_CHILD_PID=""
  fi
  exit 130
}

run_step() {
  local description="$1"
  shift

  "$@" &
  ACTIVE_CHILD_PID=$!
  set +e
  wait "$ACTIVE_CHILD_PID"
  local exit_code=$?
  set -e
  ACTIVE_CHILD_PID=""

  if [[ "$exit_code" -eq 130 ]]; then
    echo "[pretest] interrupted during $description." >&2
    exit 130
  fi
  if [[ "$exit_code" -ne 0 ]]; then
    echo "[pretest] $description failed with exit code $exit_code." >&2
    exit "$exit_code"
  fi
}

run_quiet_check() {
  "$@" >/dev/null 2>&1 &
  ACTIVE_CHILD_PID=$!
  set +e
  wait "$ACTIVE_CHILD_PID"
  local exit_code=$?
  set -e
  ACTIVE_CHILD_PID=""

  if [[ "$exit_code" -eq 130 ]]; then
    echo "[pretest] interrupted during dependency check." >&2
    exit 130
  fi
  return "$exit_code"
}

trap handle_interrupt INT TERM

resolve_python() {
  if [[ -n "$PYTHON_BIN" ]]; then
    printf '%s\n' "$PYTHON_BIN"
    return
  fi

  if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    printf '%s\n' "$PROJECT_ROOT/.venv/bin/python"
    return
  fi

  for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return
    fi
  done

  echo "[pretest] could not find a usable Python interpreter." >&2
  exit 1
}

PYTHON_BIN="$(resolve_python)"
SPLIT_DIR="$PROJECT_ROOT/data/splits/${MODE}_sub${SUBSAMPLE_N}"
GRAPH_CACHE_PATH="$SPLIT_DIR/graph_cache.pt"
RUN_NAME="preexperiment_${MODE//-/_}_esm_concat_s42_$(date +%Y%m%d_%H%M%S)"
CONFIG_PATH="$PROJECT_ROOT/config/experiments/preexperiment_esm_smoke.yaml"

cd "$PROJECT_ROOT"

if [[ "$MODE" == "cp-hard" ]]; then
  if ! run_quiet_check "$PYTHON_BIN" -c "import importlib.util, sys; has_torch = importlib.util.find_spec('torch') is not None; has_transformers = importlib.util.find_spec('transformers') is not None; sys.exit(0 if has_torch and has_transformers else 1)"; then
    echo "[pretest] cp-hard pre-experiment preparation requires torch and transformers. Run 'uv sync --extra esm' first." >&2
    exit 1
  fi
fi

echo "[pretest] preparing split files in $SPLIT_DIR"
run_step "split preparation" "$PYTHON_BIN" scripts/prepare_dti_splits.py \
  --input-path "$INPUT_PATH" \
  --output-dir "$SPLIT_DIR" \
  --mode "$MODE" \
  --seed 42 \
  --subsample-n "$SUBSAMPLE_N" \
  --build-graph-cache

DRY_RUN=0
if ! run_quiet_check "$PYTHON_BIN" -c "import torch"; then
  DRY_RUN=1
  echo "[pretest] torch is not installed. Falling back to --dry-run validation."
fi

if [[ ! -f "$GRAPH_CACHE_PATH" ]]; then
  echo "[pretest] graph cache not found at $GRAPH_CACHE_PATH"
  echo "[pretest] build it first with: $PYTHON_BIN scripts/prepare_dti_splits.py --input-path $INPUT_PATH --output-dir $SPLIT_DIR --mode $MODE --seed 42 --subsample-n $SUBSAMPLE_N --build-graph-cache"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    DRY_RUN=1
    echo "[pretest] falling back to --dry-run validation until the graph cache is available."
  fi
fi

COMMON_ARGS=(
  --config "$CONFIG_PATH"
  --set "run_name=$RUN_NAME"
  --set "data.split_name=$MODE"
  --set "data.split_dir=$SPLIT_DIR"
  --set "data.graph_cache_path=$GRAPH_CACHE_PATH"
  --set "output.allow_rerun_suffix=false"
)

TRAIN_ARGS=(-m src.train "${COMMON_ARGS[@]}")
if [[ "$DRY_RUN" -eq 1 ]]; then
  TRAIN_ARGS+=(--dry-run)
fi

echo "[pretest] launching training validation run: $RUN_NAME"
run_step "training validation run" "$PYTHON_BIN" "${TRAIN_ARGS[@]}"

if [[ "$DRY_RUN" -eq 0 && "$SKIP_EVAL" -ne 1 ]]; then
  echo "[pretest] launching evaluation validation run: $RUN_NAME"
  run_step "evaluation validation run" "$PYTHON_BIN" -m src.eval "${COMMON_ARGS[@]}"
fi
