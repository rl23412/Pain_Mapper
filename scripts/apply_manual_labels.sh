#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: bash scripts/apply_manual_labels.sh OUT_DIR RUN_TAG MANUAL_LABEL_CSV" >&2
  exit 1
fi

OUT_DIR="$1"
RUN_TAG="$2"
MANUAL_LABEL_CSV="$3"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

PYTHONPATH="$ROOT/src:${PYTHONPATH:-}" python "$ROOT/run.py" \
  --out-dir "$OUT_DIR" \
  --run-tag "$RUN_TAG" \
  --visualization-only \
  --manual-label-csv "$MANUAL_LABEL_CSV"
