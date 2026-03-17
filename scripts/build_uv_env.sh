#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-}"
RECREATE=0
FROZEN=0
EXTRAS=()

usage() {
  cat <<'EOF'
Usage: bash scripts/build_uv_env.sh [options]

Options:
  --python <bin>     Python executable or version hint to use with uv
  --extra <name>     Optional dependency extra to install (repeatable)
  --recreate         Remove the existing .venv before rebuilding
  --frozen           Require an existing uv.lock during sync
  -h, --help         Show this help message

Examples:
  bash scripts/build_uv_env.sh
  bash scripts/build_uv_env.sh --extra viz
  bash scripts/build_uv_env.sh --extra viz --extra esm
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="${2:?missing value for --python}"
      shift 2
      ;;
    --extra)
      EXTRAS+=("${2:?missing value for --extra}")
      shift 2
      ;;
    --recreate)
      RECREATE=1
      shift
      ;;
    --frozen)
      FROZEN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[uv-env] unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  echo "[uv-env] uv is not installed or not on PATH." >&2
  exit 1
fi

if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

if [[ -z "$PYTHON_BIN" ]]; then
  echo "[uv-env] could not find a usable Python interpreter." >&2
  exit 1
fi

cd "$PROJECT_ROOT"

echo "[uv-env] project_root=$PROJECT_ROOT"
echo "[uv-env] venv_dir=$VENV_DIR"
echo "[uv-env] python=$PYTHON_BIN"

if [[ "$RECREATE" -eq 1 && -d "$VENV_DIR" ]]; then
  echo "[uv-env] removing existing environment at $VENV_DIR"
  rm -rf "$VENV_DIR"
fi

uv venv --python "$PYTHON_BIN" "$VENV_DIR"

if [[ ! -f "$PROJECT_ROOT/uv.lock" && "$FROZEN" -eq 1 ]]; then
  echo "[uv-env] --frozen was requested but uv.lock does not exist." >&2
  exit 1
fi

SYNC_ARGS=(sync --python "$PYTHON_BIN")
if [[ "$FROZEN" -eq 1 ]]; then
  SYNC_ARGS+=(--frozen)
fi
if [[ "${#EXTRAS[@]}" -gt 0 ]]; then
  for extra in "${EXTRAS[@]}"; do
    SYNC_ARGS+=(--extra "$extra")
  done
fi

echo "[uv-env] running: uv ${SYNC_ARGS[*]}"
uv "${SYNC_ARGS[@]}"

echo "[uv-env] done"
echo "[uv-env] activate with: source \"$VENV_DIR/bin/activate\""
