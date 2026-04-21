#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_GLOB="${CONFIG_GLOB:-$PROJECT_ROOT/config/benchmarks/scopedti_cp-*.yaml}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/scopedti}"
GPU_IDS="${GPU_IDS:-0}"
UV_SYNC_EXTRAS="${UV_SYNC_EXTRAS:---extra train}"
RUN_ARGS=("$@")

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
  echo "[scopedti] could not find a usable Python runner." >&2
  exit 1
}

cd "$PROJECT_ROOT"
mkdir -p "$LOG_ROOT" runs/scopedti artifacts/ScopeDTI

mapfile -t CONFIGS < <(compgen -G "$CONFIG_GLOB" | sort)
if [[ "${#CONFIGS[@]}" -eq 0 ]]; then
  echo "[scopedti] no configs matched: $CONFIG_GLOB" >&2
  exit 1
fi

RUNNER_CMD="$(resolve_runner)"
echo "[scopedti] project root: $PROJECT_ROOT"
echo "[scopedti] runner: $RUNNER_CMD"
echo "[scopedti] configs: ${#CONFIGS[@]}"

for config_path in "${CONFIGS[@]}"; do
  config_name="$(basename "$config_path" .yaml)"
  log_path="$LOG_ROOT/${config_name}.log"
  echo "[scopedti] launching $config_path"
  CUDA_VISIBLE_DEVICES="$GPU_IDS" $RUNNER_CMD scripts/run_scopedti_benchmark.py \
    --config "$config_path" "${RUN_ARGS[@]}" > "$log_path" 2>&1
  echo "[scopedti] finished $config_path (log: $log_path)"
done
