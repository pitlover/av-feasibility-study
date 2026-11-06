#!/usr/bin/env bash
# Usage: bash scripts/evaluate.sh MODEL_NAME [EVAL_NAME] [infer|score|render|all]
# Requires already exported MODEL and already prepared evaluation selection.
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/evaluate.sh MODEL_NAME [EVAL_NAME] [infer|score|render|all]" >&2
  exit 2
fi
MODEL_NAME="$1"
EVAL_NAME="${2:-kitti_val_1}"
MODE="${3:-all}"
if [[ ! "$MODEL_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "[ERROR] MODEL_NAME must be a single directory name (no slashes): $MODEL_NAME" >&2
  exit 2
fi
if [[ ! "$EVAL_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "[ERROR] EVAL_NAME must be a single directory name: $EVAL_NAME" >&2
  exit 2
fi
case "$MODE" in infer|score|render|all) ;; *) echo "[ERROR] unsupported mode $MODE" >&2;exit 2;; esac
source "$(dirname "$0")/env.sh"
PYTHON="${PYTHON_BIN:-$(command -v python)}"
MODEL="$BASE/exports/$MODEL_NAME"
EVAL="$BASE/evaluation/$EVAL_NAME"
PRED="$BASE/evaluation/pred_$MODEL_NAME"
if [[ ! -f "$EVAL/selection.jsonl" ]]; then
  echo "[ERROR] Evaluation not prepared: $EVAL/selection.jsonl" >&2
  exit 1
fi
if [[ "$MODE" == infer || "$MODE" == all ]]; then
  if [[ ! -d "$MODEL" ]]; then echo "[ERROR] Exported MODEL not found: $MODEL" >&2;exit 1;fi
  if [[ ! -f "$EVAL/inference_requests.jsonl" ]]; then echo "[ERROR] Missing inference requests" >&2;exit 1;fi
  # Opt in only when necessary for this machine's TorchCodec + CUDA NPP:
  # COSMOS_PRELOAD_NPP=1 bash scripts/evaluate.sh MODEL ... infer
  if [[ "${COSMOS_PRELOAD_NPP:-0}" == "1" ]]; then
    NPP_LIBDIR="$($PYTHON -c 'from importlib.metadata import distribution; print(distribution("nvidia-npp-cu12").locate_file("nvidia/npp/lib"))')"
    test -f "$NPP_LIBDIR/libnppc.so.12" && test -f "$NPP_LIBDIR/libnppicc.so.12" || { echo '[ERROR] NPP libs missing' >&2; exit 1; }
    export LD_PRELOAD="$NPP_LIBDIR/libnppc.so.12:$NPP_LIBDIR/libnppicc.so.12${LD_PRELOAD:+:$LD_PRELOAD}"
  fi
  echo "[INFER] model=$MODEL, input=$EVAL, output=$PRED"
  "$PYTHON" -m cosmos3eval.cli infer \
    --eval-dir "$EVAL" --pred-dir "$PRED" \
    --framework-root "$FRAMEWORK" --checkpoint "$MODEL" --gpus "${GPUS_INFER:-1}" --execute
fi
if [[ "$MODE" == score || "$MODE" == all ]]; then
  echo "[SCORE] $PRED"
  "$PYTHON" -m cosmos3eval.cli score --eval-dir "$EVAL" --pred-dir "$PRED" --model "$MODEL_NAME"
fi
if [[ "$MODE" == render || "$MODE" == all ]]; then
  echo "[RENDER] $PRED/qualitative"
  "$PYTHON" -m cosmos3eval.cli render --eval-dir "$EVAL" --pred-dir "$PRED" \
    --out-dir "$PRED/qualitative" --max-clips "${RENDER_MAX_CLIPS:-1}"
fi
