#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_GLOB="${CONFIG_GLOB:-$PROJECT_ROOT/config/experiments/exprS*.yaml}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/exprS}"
STATE_ROOT="${STATE_ROOT:-$LOG_ROOT/.run_exprS_matrix_ubuntu}"
RESULT_ROOT="${RESULT_ROOT:-/root/autodl-tmp/results}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
TRAIN_MODULE="${TRAIN_MODULE:-src.train}"
UV_SYNC_EXTRAS="${UV_SYNC_EXTRAS:---extra train --extra esm --extra esmc}"
RUNNER_SPEC="${RUNNER:-}"
INTERRUPTED=0
TRAIN_ARGS=()
TRAIN_ARGS_COUNT=0
declare -a DEFAULT_TRAIN_ARGS=()
DEFAULT_TRAIN_ARGS_COUNT=0

declare -a GPU_LIST=()
declare -a CONFIGS=()
declare -a RUNNER_CMD=()
declare -a UV_SYNC_EXTRA_ARGS=()
declare -a SLOT_PIDS=()
declare -a SLOT_CONFIGS=()
declare -a SLOT_STATUS_FILES=()
RUNNER_LABEL=""
RUN_STATUS=0
LAUNCH_PID=""

usage() {
  cat <<'EOF'
Usage: bash scripts/run_exprS_matrix_ubuntu.sh [options] [-- <train args...>]

Run all exprS experiment configs across multiple GPUs on Ubuntu/Linux by
scheduling one training process per visible GPU.

Options:
  --config-glob <glob>    Config glob to expand
                          default: config/experiments/exprS*.yaml
  --log-root <dir>        Directory for per-config stdout/stderr logs
                          default: logs/exprS
  --state-root <dir>      Directory for scheduler status files
                          default: <log-root>/.run_exprS_matrix_ubuntu
  --result-root <dir>     Default output.root_dir for training runs
                          default: /root/autodl-tmp/results
  --gpu-ids <csv>         Comma-separated GPU ids, e.g. 0,1,2,3
                          default: 0,1,2,3
  --runner <cmd>          Override Python runner command, e.g. ".venv/bin/python"
  --train-module <name>   Python module to execute
                          default: src.train
  --uv-sync-extras <arg>  Args passed to uv sync/uv run when uv is used
                          default: --extra train --extra esm --extra esmc
  -h, --help              Show this help message

Environment overrides:
  CONFIG_GLOB, LOG_ROOT, STATE_ROOT, RESULT_ROOT, GPU_IDS, RUNNER, TRAIN_MODULE,
  UV_SYNC_EXTRAS

Examples:
  bash scripts/run_exprS_matrix_ubuntu.sh
  bash scripts/run_exprS_matrix_ubuntu.sh --gpu-ids 0,1
  bash scripts/run_exprS_matrix_ubuntu.sh --config-glob 'config/experiments/exprS2_*.yaml'
  bash scripts/run_exprS_matrix_ubuntu.sh --result-root /root/autodl-tmp/results
  bash scripts/run_exprS_matrix_ubuntu.sh -- --seed 42
EOF
}

trim_whitespace() {
  local value="$1"

  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s\n' "$value"
}

join_shell_words() {
  local -a quoted=()
  local arg

  for arg in "$@"; do
    quoted+=("$(printf '%q' "$arg")")
  done

  printf '%s' "${quoted[*]:-}"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --config-glob)
        CONFIG_GLOB="${2:?missing value for --config-glob}"
        shift 2
        ;;
      --log-root)
        LOG_ROOT="${2:?missing value for --log-root}"
        shift 2
        ;;
      --state-root)
        STATE_ROOT="${2:?missing value for --state-root}"
        shift 2
        ;;
      --result-root)
        RESULT_ROOT="${2:?missing value for --result-root}"
        shift 2
        ;;
      --gpu-ids)
        GPU_IDS="${2:?missing value for --gpu-ids}"
        shift 2
        ;;
      --runner)
        RUNNER_SPEC="${2:?missing value for --runner}"
        shift 2
        ;;
      --train-module)
        TRAIN_MODULE="${2:?missing value for --train-module}"
        shift 2
        ;;
      --uv-sync-extras)
        UV_SYNC_EXTRAS="${2:?missing value for --uv-sync-extras}"
        shift 2
        ;;
      --)
        shift
        TRAIN_ARGS=("$@")
        TRAIN_ARGS_COUNT="$#"
        return
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        TRAIN_ARGS=("$@")
        TRAIN_ARGS_COUNT="$#"
        return
        ;;
    esac
  done
}

parse_gpu_ids() {
  local -a raw_values=()
  local value
  local trimmed

  GPU_LIST=()
  if [[ -z "$GPU_IDS" ]]; then
    return
  fi

  IFS=',' read -r -a raw_values <<< "$GPU_IDS"
  for value in "${raw_values[@]}"; do
    trimmed="$(trim_whitespace "$value")"
    if [[ -n "$trimmed" ]]; then
      GPU_LIST+=("$trimmed")
    fi
  done
}

parse_uv_sync_extras() {
  UV_SYNC_EXTRA_ARGS=()
  if [[ -n "$UV_SYNC_EXTRAS" ]]; then
    read -r -a UV_SYNC_EXTRA_ARGS <<< "$UV_SYNC_EXTRAS"
  fi
}

prepare_default_train_args() {
  DEFAULT_TRAIN_ARGS=(--set "output.root_dir=$RESULT_ROOT")
  DEFAULT_TRAIN_ARGS_COUNT="${#DEFAULT_TRAIN_ARGS[@]}"
}

prepare_runner() {
  local candidate

  parse_uv_sync_extras

  if [[ -n "$RUNNER_SPEC" ]]; then
    read -r -a RUNNER_CMD <<< "$RUNNER_SPEC"
    if [[ "${#RUNNER_CMD[@]}" -eq 0 ]]; then
      echo "[exprS] --runner did not contain an executable." >&2
      exit 1
    fi
    RUNNER_LABEL="$(join_shell_words "${RUNNER_CMD[@]}")"
    return
  fi

  if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    RUNNER_CMD=("$PROJECT_ROOT/.venv/bin/python")
    RUNNER_LABEL="$(join_shell_words "${RUNNER_CMD[@]}")"
    return
  fi

  if command -v uv >/dev/null 2>&1; then
    echo "[exprS] .venv/bin/python not found. Preparing environment with uv sync ${UV_SYNC_EXTRAS}" >&2
    uv sync "${UV_SYNC_EXTRA_ARGS[@]}"

    if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
      RUNNER_CMD=("$PROJECT_ROOT/.venv/bin/python")
      RUNNER_LABEL="$(join_shell_words "${RUNNER_CMD[@]}")"
      return
    fi

    RUNNER_CMD=(uv run --no-sync "${UV_SYNC_EXTRA_ARGS[@]}" python)
    RUNNER_LABEL="$(join_shell_words "${RUNNER_CMD[@]}")"
    return
  fi

  for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      RUNNER_CMD=("$candidate")
      RUNNER_LABEL="$(join_shell_words "${RUNNER_CMD[@]}")"
      echo "[exprS] using fallback runner: $candidate" >&2
      return
    fi
  done

  echo "[exprS] could not find a usable Python runner." >&2
  echo "[exprS] create the environment with: uv sync --extra train --extra esm --extra esmc" >&2
  exit 1
}

collect_configs() {
  mapfile -t CONFIGS < <(compgen -G "$CONFIG_GLOB" | LC_ALL=C sort)
}

init_slots() {
  local idx

  SLOT_PIDS=()
  SLOT_CONFIGS=()
  SLOT_STATUS_FILES=()
  RUN_STATUS=0

  for idx in "${!GPU_LIST[@]}"; do
    SLOT_PIDS[$idx]=""
    SLOT_CONFIGS[$idx]=""
    SLOT_STATUS_FILES[$idx]=""
  done
}

finalize_slot() {
  local idx="$1"
  local status_code="$2"
  local status_file="${SLOT_STATUS_FILES[$idx]:-}"

  if [[ "$status_code" -ne 0 ]]; then
    echo "[exprS] ${SLOT_CONFIGS[$idx]:-unknown} exited with status $status_code" >&2
    RUN_STATUS=1
  fi

  if [[ -n "$status_file" ]]; then
    rm -f "$status_file"
  fi

  SLOT_PIDS[$idx]=""
  SLOT_CONFIGS[$idx]=""
  SLOT_STATUS_FILES[$idx]=""
}

handle_interrupt() {
  local signal_name="${1:-INT}"
  local idx
  local pid

  if [[ "$INTERRUPTED" -eq 1 ]]; then
    return
  fi
  INTERRUPTED=1

  echo "[exprS] received ${signal_name}. Stopping active jobs..." >&2
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
    if [[ -n "${SLOT_STATUS_FILES[$idx]:-}" ]]; then
      rm -f "${SLOT_STATUS_FILES[$idx]}"
      SLOT_STATUS_FILES[$idx]=""
    fi
  done

  exit 130
}

refresh_slots() {
  local idx
  local pid
  local status_file
  local status_code

  for idx in "${!GPU_LIST[@]}"; do
    pid="${SLOT_PIDS[$idx]:-}"
    status_file="${SLOT_STATUS_FILES[$idx]:-}"
    if [[ -z "$pid" ]]; then
      continue
    fi

    if [[ -n "$status_file" && -f "$status_file" ]]; then
      status_code="$(<"$status_file")"
      set +e
      wait "$pid"
      set -e
      finalize_slot "$idx" "$status_code"
      continue
    fi

    if ! kill -0 "$pid" >/dev/null 2>&1; then
      set +e
      wait "$pid"
      status_code=$?
      set -e
      if [[ -n "$status_file" && -f "$status_file" ]]; then
        status_code="$(<"$status_file")"
      fi
      finalize_slot "$idx" "$status_code"
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
    if [[ -z "${SLOT_PIDS[$idx]:-}" ]]; then
      printf '%s\n' "$idx"
      return
    fi
  done
  printf '%s\n' "-1"
}

wait_for_slot() {
  while true; do
    refresh_slots
    if has_free_slot; then
      return
    fi
    sleep 5
  done
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

    sleep 5
  done
}

launch_job() {
  local gpu_id="$1"
  local config_path="$2"
  local config_name="$3"
  local log_path="$4"
  local status_path="$5"
  local extra_args_label=""
  local -a train_command=()

  rm -f "$status_path"
  if [[ "$DEFAULT_TRAIN_ARGS_COUNT" -gt 0 && "$TRAIN_ARGS_COUNT" -gt 0 ]]; then
    extra_args_label="$(join_shell_words "${DEFAULT_TRAIN_ARGS[@]}" "${TRAIN_ARGS[@]}")"
  elif [[ "$DEFAULT_TRAIN_ARGS_COUNT" -gt 0 ]]; then
    extra_args_label="$(join_shell_words "${DEFAULT_TRAIN_ARGS[@]}")"
  elif [[ "$TRAIN_ARGS_COUNT" -gt 0 ]]; then
    extra_args_label="$(join_shell_words "${TRAIN_ARGS[@]}")"
  fi

  train_command=("${RUNNER_CMD[@]}" -m "$TRAIN_MODULE" --config "$config_path")
  if [[ "$DEFAULT_TRAIN_ARGS_COUNT" -gt 0 ]]; then
    train_command+=("${DEFAULT_TRAIN_ARGS[@]}")
  fi
  if [[ "$TRAIN_ARGS_COUNT" -gt 0 ]]; then
    train_command+=("${TRAIN_ARGS[@]}")
  fi

  (
    local exit_code

    if ! cd "$PROJECT_ROOT"; then
      exit_code=$?
      printf '%s\n' "$exit_code" >"$status_path"
      exit "$exit_code"
    fi

    export CUDA_VISIBLE_DEVICES="$gpu_id"

    echo "[exprS] host: $(hostname)"
    echo "[exprS] start: $(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "[exprS] config: $config_path"
    echo "[exprS] run: $config_name"
    echo "[exprS] gpu: $gpu_id"
    echo "[exprS] runner: $RUNNER_LABEL"
    echo "[exprS] train module: $TRAIN_MODULE"
    if [[ -n "$extra_args_label" ]]; then
      echo "[exprS] extra args: $extra_args_label"
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
      echo "[exprS] nvidia-smi"
      nvidia-smi || true
    fi

    set +e
    "${train_command[@]}"
    exit_code=$?
    set -e

    echo "[exprS] end: $(date '+%Y-%m-%d %H:%M:%S %z')"
    printf '%s\n' "$exit_code" >"$status_path"
    exit "$exit_code"
  ) >"$log_path" 2>&1 &

  LAUNCH_PID=$!
}

parse_args "$@"

cd "$PROJECT_ROOT"
mkdir -p "$LOG_ROOT" "$STATE_ROOT"

prepare_runner
parse_gpu_ids
prepare_default_train_args

if [[ "${#GPU_LIST[@]}" -eq 0 ]]; then
  echo "[exprS] GPU_IDS must list at least one CUDA device." >&2
  exit 1
fi

collect_configs
if [[ "${#CONFIGS[@]}" -eq 0 ]]; then
  echo "[exprS] no configs matched: $CONFIG_GLOB" >&2
  exit 1
fi

echo "[exprS] project root: $PROJECT_ROOT"
echo "[exprS] runner: $RUNNER_LABEL"
echo "[exprS] gpu ids: ${GPU_LIST[*]}"
echo "[exprS] configs: ${#CONFIGS[@]}"
echo "[exprS] log root: $LOG_ROOT"
echo "[exprS] result root: $RESULT_ROOT"
echo "[exprS] default train args: ${DEFAULT_TRAIN_ARGS[*]}"
if [[ "$TRAIN_ARGS_COUNT" -gt 0 ]]; then
  echo "[exprS] extra train args: ${TRAIN_ARGS[*]}"
fi

init_slots
trap 'handle_interrupt INT' INT
trap 'handle_interrupt TERM' TERM

for config_path in "${CONFIGS[@]}"; do
  wait_for_slot
  slot_index="$(find_free_gpu_slot)"
  if [[ "$slot_index" == "-1" ]]; then
    echo "[exprS] internal scheduling error: no free GPU slot found." >&2
    exit 1
  fi

  gpu_id="${GPU_LIST[$slot_index]}"
  config_name="$(basename "$config_path" .yaml)"
  log_path="$LOG_ROOT/${config_name}.log"
  status_path="$STATE_ROOT/${config_name}.status"

  echo "[exprS] launching $config_name on cuda:$gpu_id"
  launch_job "$gpu_id" "$config_path" "$config_name" "$log_path" "$status_path"
  SLOT_PIDS[$slot_index]="$LAUNCH_PID"
  SLOT_CONFIGS[$slot_index]="$config_name"
  SLOT_STATUS_FILES[$slot_index]="$status_path"
done

if wait_all; then
  echo "[exprS] all jobs completed successfully."
  exit 0
fi

echo "[exprS] one or more jobs failed. Inspect logs under $LOG_ROOT." >&2
exit 1
