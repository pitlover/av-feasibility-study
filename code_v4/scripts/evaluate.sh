#!/usr/bin/env bash
# Usage: bash scripts/evaluate.sh MODEL_NAME EVAL_NAME [infer|score|render|all]
set -euo pipefail
[[ $# -ge 2 ]] || { echo "Usage: $0 MODEL_NAME EVAL_NAME [infer|score|render|all]" >&2; exit 2; }
MODEL_NAME="$1";EVAL_NAME="$2";MODE="${3:-all}"
[[ "$MODEL_NAME" =~ ^[A-Za-z0-9._-]+$ && "$EVAL_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo Invalid names >&2; exit 2; }
case "$MODE" in infer|score|render|all) ;; *) echo Invalid mode >&2; exit 2 ;; esac
source "$(dirname "$0")/env.sh"
PYTHON="${PYTHON_BIN:-$(command -v python)}"
MODEL="$BASE/exports/$MODEL_NAME"
EVAL="$BASE/evaluation/$EVAL_NAME"
PRED="$BASE/evaluation/pred_$MODEL_NAME"
[[ -f "$EVAL/selection.jsonl" ]] || { echo "Missing eval selection $EVAL" >&2; exit 1; }
if [[ "$MODE" = infer || "$MODE" = all ]]; then
    [[ -d "$MODEL" && -f "$EVAL/inference_requests.jsonl" ]] || { echo Missing model or inference requests >&2; exit 1; }
    "$PYTHON" -m cosmos3eval.cli infer --eval-dir "$EVAL" --pred-dir "$PRED" \
      --framework-root "$FRAMEWORK" --checkpoint "$MODEL" --gpus "${GPUS_INFER:-1}" --execute
fi
if [[ "$MODE" = score || "$MODE" = all ]]; then
    "$PYTHON" -m cosmos3eval.cli score --eval-dir "$EVAL" --pred-dir "$PRED" --model "$MODEL_NAME"
fi
if [[ "$MODE" = render || "$MODE" = all ]]; then
    "$PYTHON" -m cosmos3eval.cli render --eval-dir "$EVAL" --pred-dir "$PRED" \
      --out-dir "$PRED/qualitative" --max-clips "${RENDER_MAX_CLIPS:-1}"
fi
