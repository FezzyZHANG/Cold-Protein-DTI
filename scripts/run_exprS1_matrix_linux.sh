#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_GLOB="${CONFIG_GLOB:-$PROJECT_ROOT/config/experiments/exprS1_*.yaml}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/exprS1}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
TRAIN_MODULE="${TRAIN_MODULE:-src.train}"
UV_SYNC_EXTRAS="${UV_SYNC_EXTRAS:---extra train --extra esm}"

resolve_runner() {
  if [[ -n "${RUNNER:-}" ]]; then
    printf '%s\n' "$RUNNER"
    return
  fi

  if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    printf '%s\n' "$PROJECT_ROOT/.venv/bin/python"
    return
  fi

  if command -v uv >/dev/null 2>&1; then
    printf 'uv run %s python\n' "$UV_SYNC_EXTRAS"
    return
  fi

  for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return
    fi
  done

  echo "[exprS1] could not find a usable Python runner." >&2
  echo "[exprS1] create the environment with: uv sync --extra train --extra esm" >&2
  exit 1
}

split_csv() {
  local input="$1"
  local output_var="$2"
  local -a parsed_values=()
  IFS=',' read -r -a parsed_values <<< "$input"
  eval "$output_var=(\"\${parsed_values[@]}\")"
}

wait_for_slot() {
  local running_count
  while true; do
    running_count=0
    for pid in "${SLOT_PIDS[@]-}"; do
      if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        running_count=$((running_count + 1))
      fi
    done
    if [[ "$running_count" -lt "${#GPU_LIST[@]}" ]]; then
      return
    fi
    sleep 5
  done
}

find_free_gpu_slot() {
  local idx
  for idx in "${!GPU_LIST[@]}"; do
    local pid="${SLOT_PIDS[$idx]:-}"
    if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
      echo "$idx"
      return
    fi
  done
  echo "-1"
}

wait_all() {
  local status=0
  local pid
  for pid in "${ALL_PIDS[@]-}"; do
    if [[ -n "$pid" ]]; then
      if ! wait "$pid"; then
        status=1
      fi
    fi
  done
  return "$status"
}

cd "$PROJECT_ROOT"
mkdir -p "$LOG_ROOT"

RUNNER_CMD="$(resolve_runner)"
split_csv "$GPU_IDS" GPU_LIST

if [[ "${#GPU_LIST[@]}" -eq 0 ]]; then
  echo "[exprS1] GPU_IDS must list at least one CUDA device." >&2
  exit 1
fi

mapfile -t CONFIGS < <(compgen -G "$CONFIG_GLOB" | sort)
if [[ "${#CONFIGS[@]}" -eq 0 ]]; then
  echo "[exprS1] no configs matched: $CONFIG_GLOB" >&2
  exit 1
fi

echo "[exprS1] project root: $PROJECT_ROOT"
echo "[exprS1] runner: $RUNNER_CMD"
echo "[exprS1] gpu ids: ${GPU_LIST[*]}"
echo "[exprS1] configs: ${#CONFIGS[@]}"

declare -a SLOT_PIDS=()
declare -a ALL_PIDS=()

for config_path in "${CONFIGS[@]}"; do
  wait_for_slot
  slot_index="$(find_free_gpu_slot)"
  if [[ "$slot_index" == "-1" ]]; then
    echo "[exprS1] internal scheduling error: no free GPU slot found." >&2
    exit 1
  fi

  gpu_id="${GPU_LIST[$slot_index]}"
  config_name="$(basename "$config_path" .yaml)"
  log_path="$LOG_ROOT/${config_name}.log"

  echo "[exprS1] launching $config_name on cuda:$gpu_id"
  CUDA_VISIBLE_DEVICES="$gpu_id" bash -lc \
    "cd \"$PROJECT_ROOT\" && $RUNNER_CMD -m $TRAIN_MODULE --config \"$config_path\"" \
    >"$log_path" 2>&1 &
  SLOT_PIDS[$slot_index]=$!
  ALL_PIDS+=("${SLOT_PIDS[$slot_index]}")
done

if wait_all; then
  echo "[exprS1] all jobs completed successfully."
  exit 0
fi

echo "[exprS1] one or more jobs failed. Inspect logs under $LOG_ROOT." >&2
exit 1
