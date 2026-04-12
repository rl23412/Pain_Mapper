#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/run_fixed_pipeline.sh DATA_ROOT OUT_DIR [RUN_TAG]" >&2
  exit 1
fi

DATA_ROOT="$1"
OUT_DIR="$2"
RUN_TAG="${3:-population_cap_formalin_joint}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SELECTED_MNN_K="${SELECTED_MNN_K:-20}"
SELECTED_TSNE_PERPLEXITY="${SELECTED_TSNE_PERPLEXITY:-32}"
TSNE_BACKEND="${TSNE_BACKEND:-pca}"
PIPELINE_PRESET="${PIPELINE_PRESET:-}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

ARGS=(
  --data-root "$DATA_ROOT"
  --out-dir "$OUT_DIR"
  --run-tag "$RUN_TAG"
  --selected-mnn-k "$SELECTED_MNN_K"
  --selected-tsne-perplexity "$SELECTED_TSNE_PERPLEXITY"
  --tsne-backend "$TSNE_BACKEND"
)

if [[ -n "$PIPELINE_PRESET" ]]; then
  ARGS+=(--preset "$PIPELINE_PRESET")
fi

PYTHONPATH="$ROOT/src:${PYTHONPATH:-}" python "$ROOT/run.py" "${ARGS[@]}"
