#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_GLOB="${CONFIG_GLOB:-$PROJECT_ROOT/config/experiments/exprS2_*.yaml}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/exprS2}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
TRAIN_MODULE="${TRAIN_MODULE:-src.train}"
UV_SYNC_EXTRAS="${UV_SYNC_EXTRAS:---extra train --extra esm}"
INTERRUPTED=0

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

  echo "[exprS2] could not find a usable Python runner." >&2
  echo "[exprS2] create the environment with: uv sync --extra train --extra esm" >&2
  exit 1
}

split_csv() {
  local input="$1"
  local output_var="$2"
  local -a parsed_values=()
  IFS=',' read -r -a parsed_values <<< "$input"
  eval "$output_var=(\"\${parsed_values[@]}\")"
}

init_slots() {
  local idx

  SLOT_PIDS=()
  SLOT_CONFIGS=()
  RUN_STATUS=0

  for idx in "${!GPU_LIST[@]}"; do
    SLOT_PIDS[$idx]=""
    SLOT_CONFIGS[$idx]=""
  done
}

handle_interrupt() {
  local signal_name="${1:-INT}"
  local idx
  local pid

  if [[ "$INTERRUPTED" -eq 1 ]]; then
    return
  fi
  INTERRUPTED=1

  echo "[exprS2] received ${signal_name}. Stopping active jobs..." >&2
  for idx in "${!GPU_LIST[@]}"; do
    pid="${SLOT_PIDS[$idx]:-}"
    if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
      kill -INT "$pid" >/dev/null 2>&1 || true
    fi
  done

  for idx in "${!GPU_LIST[@]}"; do
    pid="${SLOT_PIDS[$idx]:-}"
    if [[ -n "$pid" ]]; then
      set +e
      wait "$pid"
      set -e
      SLOT_PIDS[$idx]=""
      SLOT_CONFIGS[$idx]=""
    fi
  done

  exit 130
}

wait_for_slot() {
  while true; do
    refresh_slots
    if has_free_slot; then
      return
    fi
    # echo "[exprS2] no free GPU slots available. Waiting..." >&2
    sleep 5
  done
}

refresh_slots() {
  local idx
  local pid
  local running_pid
  local is_running
  local -a running_pids=()

  mapfile -t running_pids < <(jobs -pr)

  for idx in "${!GPU_LIST[@]}"; do
    pid="${SLOT_PIDS[$idx]:-}"
    if [[ -z "$pid" ]]; then
      continue
    fi

    is_running=0
    for running_pid in "${running_pids[@]}"; do
      if [[ "$running_pid" == "$pid" ]]; then
        is_running=1
        break
      fi
    done

    if [[ "$is_running" -eq 0 ]]; then
      if ! wait "$pid"; then
        RUN_STATUS=1
      fi
      SLOT_PIDS[$idx]=""
      SLOT_CONFIGS[$idx]=""
    fi
  done
}

has_free_slot() {
  local idx
  for idx in "${!GPU_LIST[@]}"; do
    if [[ -z "${SLOT_PIDS[$idx]:-}" ]]; then
      return 0
    fi
  done
  return 1
}

find_free_gpu_slot() {
  local idx
  for idx in "${!GPU_LIST[@]}"; do
    local pid="${SLOT_PIDS[$idx]:-}"
    if [[ -z "$pid" ]]; then
      echo "$idx"
      return
    fi
  done
  echo "-1"
}

wait_all() {
  while true; do
    local idx
    local any_running=0

    refresh_slots

    for idx in "${!GPU_LIST[@]}"; do
      if [[ -n "${SLOT_PIDS[$idx]:-}" ]]; then
        any_running=1
        break
      fi
    done

    if [[ "$any_running" -eq 0 ]]; then
      return "$RUN_STATUS"
    fi

    echo "[exprS2] waiting for ${#GPU_LIST[@]} GPU slots to be free..." >&2
    sleep 5
  done
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

echo "[exprS2] project root: $PROJECT_ROOT"
echo "[exprS2] runner: $RUNNER_CMD"
echo "[exprS2] gpu ids: ${GPU_LIST[*]}"
echo "[exprS2] configs: ${#CONFIGS[@]}"

declare -a SLOT_PIDS=()
declare -a SLOT_CONFIGS=()
init_slots
trap 'handle_interrupt INT' INT
trap 'handle_interrupt TERM' TERM

for config_path in "${CONFIGS[@]}"; do
  wait_for_slot
  slot_index="$(find_free_gpu_slot)"
  if [[ "$slot_index" == "-1" ]]; then
    echo "[exprS2] internal scheduling error: no free GPU slot found." >&2
    exit 1
  fi

  gpu_id="${GPU_LIST[$slot_index]}"
  config_name="$(basename "$config_path" .yaml)"
  log_path="$LOG_ROOT/${config_name}.log"

  echo "[exprS2] launching $config_name on cuda:$gpu_id"
  CUDA_VISIBLE_DEVICES="$gpu_id" bash -lc \
    "cd \"$PROJECT_ROOT\" && exec $RUNNER_CMD -m $TRAIN_MODULE --config \"$config_path\"" \
    >"$log_path" 2>&1 &
  SLOT_PIDS[$slot_index]=$!
  SLOT_CONFIGS[$slot_index]="$config_name"
done

if wait_all; then
  echo "[exprS2] all jobs completed successfully."
  exit 0
fi

echo "[exprS2] one or more jobs failed. Inspect logs under $LOG_ROOT." >&2
exit 1
