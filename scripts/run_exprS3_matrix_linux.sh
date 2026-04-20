#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export CONFIG_GLOB="${CONFIG_GLOB:-$PROJECT_ROOT/config/experiments/exprS3_*.yaml}"
export LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/exprS3}"
export UV_SYNC_EXTRAS="${UV_SYNC_EXTRAS:---extra train --extra esm --extra esmc}"

exec bash "$PROJECT_ROOT/scripts/run_exprS_matrix_ubuntu.sh" "$@"
