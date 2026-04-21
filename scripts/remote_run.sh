#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
EXP_PATH="${2:-}"
shift $(( $# >= 2 ? 2 : $# ))
EXTRA_ARGS=("$@")

if [[ -z "$MODE" || -z "$EXP_PATH" ]]; then
  echo "Usage: bash scripts/remote_run.sh <train|eval|kdbnet|scopedti> <config.yaml>"
  exit 1
fi

if [[ ! -f "$EXP_PATH" ]]; then
  echo "Config not found: $EXP_PATH"
  exit 1
fi

# Optional environment activation.
if [[ -n "${REMOTE_ACTIVATE:-}" ]]; then
  # shellcheck disable=SC1090
  eval "$REMOTE_ACTIVATE"
fi

echo "[INFO] Host: $(hostname)"
echo "[INFO] Date: $(date)"
echo "[INFO] Mode: $MODE"
echo "[INFO] Config: $EXP_PATH"
echo "[INFO] Python: $(which python || true)"
python -V

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[INFO] nvidia-smi summary"
  nvidia-smi || true
fi

handle_interrupt() {
  echo "[INFO] interrupt received. Exiting remote run wrapper." >&2
  exit 130
}

trap handle_interrupt INT TERM

case "$MODE" in
  train)
    exec python -m src.train --config "$EXP_PATH" "${EXTRA_ARGS[@]}"
    ;;
  eval)
    exec python -m src.eval --config "$EXP_PATH" "${EXTRA_ARGS[@]}"
    ;;
  kdbnet)
    exec python scripts/run_kdbnet_benchmark.py --config "$EXP_PATH" "${EXTRA_ARGS[@]}"
    ;;
  scopedti)
    exec python scripts/run_scopedti_benchmark.py --config "$EXP_PATH" "${EXTRA_ARGS[@]}"
    ;;
  *)
    echo "Unknown mode: $MODE"
    exit 1
    ;;
esac
