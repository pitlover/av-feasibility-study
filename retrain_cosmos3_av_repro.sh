#!/usr/bin/env bash
# Reproducible Cosmos3-Edge AV-ID experiment; keeps all existing training runs intact.
# Place at the cosmos3_ji project root. Usage:
#   bash retrain_cosmos3_av_repro.sh stage1
#   bash retrain_cosmos3_av_repro.sh stage2
#   bash retrain_cosmos3_av_repro.sh export
#   bash retrain_cosmos3_av_repro.sh evaluate
# or: bash retrain_cosmos3_av_repro.sh all
# Configure RUN_TAG, MAX_ITERS, GPUS, EPOCH_CLIPS through environment if necessary.
set -Eeuo pipefail

MODE="${1:-stage1}"
BASE="${BASE:-$(pwd)}"
BASE="$(cd "$BASE" && pwd)"
FRAMEWORK="${FRAMEWORK:-$BASE/cosmos-framework}"
PROCESSED="${PROCESSED:-$BASE/processed_virtual_native_v2}"
TEMPLATE_RUN="${TEMPLATE_RUN:-$BASE/runs_virtual/stage1_joint_300}"
RUN_TAG="${RUN_TAG:-repro_20261108_a}"
GPUS="${GPUS:-4}"
MAX_ITERS="${MAX_ITERS:-300}"
SAVE_EVERY="${SAVE_EVERY:-50}"
EPOCH_CLIPS="${EPOCH_CLIPS:-500}"
EVAL_CLIPS="${EVAL_CLIPS:-5}"
SEED="${SEED:-42}"
# Include the previously diagnosed AV2 clip in val evaluation when it is a val clip.
MATCH_CLIP_ID="${MATCH_CLIP_ID:-av2_02678d04-cc9f-3148-9f95-1ba66347dff9_000000}"

S1="$BASE/runs_repro/stage1_joint_${MAX_ITERS}_${RUN_TAG}"
S2="$BASE/runs_repro/stage2_av2_${MAX_ITERS}_${RUN_TAG}"
S1_CKPT="$S1/outputs/cosmos3_av_id/pretrain/$(basename "$S1")/checkpoints/$(printf 'iter_%09d' "$MAX_ITERS")"
S2_CKPT="$S2/outputs/cosmos3_av_id/posttrain/$(basename "$S2")/checkpoints/$(printf 'iter_%09d' "$MAX_ITERS")"
S1_CONFIG="$S1/outputs/cosmos3_av_id/pretrain/$(basename "$S1")/config.yaml"
S2_CONFIG="$S2/outputs/cosmos3_av_id/posttrain/$(basename "$S2")/config.yaml"
MODEL1="$BASE/exports/$(basename "$S1")"
MODEL2="$BASE/exports/$(basename "$S2")"
EVAL="$BASE/evaluation/av2_val_${RUN_TAG}"
PRED1="$BASE/evaluation/pred_$(basename "$S1")"
PRED2="$BASE/evaluation/pred_$(basename "$S2")"
export PYTHONPATH="$BASE:$FRAMEWORK${PYTHONPATH:+:$PYTHONPATH}"

fatal(){ echo "[ERROR] $*" >&2; exit 1; }
require_file(){ [[ -f "$1" ]] || fatal "Missing file: $1"; }
require_ckpt(){ require_file "$1/model/.metadata"; require_file "$1/trainer/.metadata"; }

require_file "$TEMPLATE_RUN/run.json"
require_file "$PROCESSED/index.jsonl"
require_file "$FRAMEWORK/cosmos_framework/scripts/train.py"
EDGE_DCP="$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["init_checkpoint"])' "$TEMPLATE_RUN/run.json")"
WAN_VAE="$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["wan_vae"])' "$TEMPLATE_RUN/run.json")"
[[ -d "$EDGE_DCP" ]] || fatal "Cosmos3-Edge checkpoint missing: $EDGE_DCP"
require_file "$WAN_VAE"

print_config(){
  printf '\n[CONFIG] mode=%s run_tag=%s gpus=%s iters=%s\n' "$MODE" "$RUN_TAG" "$GPUS" "$MAX_ITERS"
  printf '[CONFIG] BASE=%s\n[CONFIG] PROCESSED=%s\n' "$BASE" "$PROCESSED"
  printf '[CONFIG] EDGE_DCP=%s\n[CONFIG] WAN_VAE=%s\n' "$EDGE_DCP" "$WAN_VAE"
  printf '[CONFIG] Stage1=%s\n[CONFIG] Stage2=%s\n' "$S1" "$S2"
}

stage1(){
  [[ ! -e "$S1" ]] || fatal "Stage1 run already exists; use a new RUN_TAG to avoid overwriting: $S1"
  echo '[STAGE1] Creating KITTI+Waymo pretraining plan'
  python -m cosmos3av.cli plan \
    --stage pretrain --datasets kitti waymo --weights kitti:1,waymo:1 \
    --processed-root "$PROCESSED" --init-checkpoint "$EDGE_DCP" --wan-vae "$WAN_VAE" \
    --outdir "$S1" --epoch-clips "$EPOCH_CLIPS" --max-iters "$MAX_ITERS" \
    --save-every "$SAVE_EVERY" --gpus "$GPUS" --seed "$SEED" --full-preflight
  echo '[STAGE1] Running dataset/model sample contract check'
  python -m cosmos3av.cli smoke --manifest "$S1/train.jsonl"
  echo '[STAGE1] Training starts'
  unset IMAGINAIRE_OUTPUT_ROOT || true
  python -m cosmos3av.cli launch --run-dir "$S1" --framework-root "$FRAMEWORK" --execute \
    2>&1 | tee "$S1/train.log"
  require_ckpt "$S1_CKPT"
  echo "[STAGE1 OK] $S1_CKPT"
}

stage2(){
  require_ckpt "$S1_CKPT"
  [[ ! -e "$S2" ]] || fatal "Stage2 run already exists; use a new RUN_TAG: $S2"
  echo '[STAGE2] Creating AV2 posttraining plan initialized from Stage1 DCP'
  python -m cosmos3av.cli plan \
    --stage posttrain --datasets av2 --stage1-run "$S1" \
    --processed-root "$PROCESSED" --init-checkpoint "$S1_CKPT" --wan-vae "$WAN_VAE" \
    --outdir "$S2" --epoch-clips "$EPOCH_CLIPS" --max-iters "$MAX_ITERS" \
    --save-every "$SAVE_EVERY" --gpus "$GPUS" --seed "$SEED" --full-preflight
  echo '[STAGE2] Running dataset/model sample contract check'
  python -m cosmos3av.cli smoke --manifest "$S2/train.jsonl"
  echo '[STAGE2] Training starts'
  unset IMAGINAIRE_OUTPUT_ROOT || true
  python -m cosmos3av.cli launch --run-dir "$S2" --framework-root "$FRAMEWORK" --execute \
    2>&1 | tee "$S2/train.log"
  require_ckpt "$S2_CKPT"
  echo "[STAGE2 OK] $S2_CKPT"
}

export_one(){
  local ckpt="$1" cfg="$2" run="$3" model="$4" init="$5"
  require_ckpt "$ckpt"
  require_file "$cfg"
  [[ ! -e "$model" ]] || fatal "Export output already exists: $model; use another RUN_TAG"
  export COSMOS_AV_MANIFEST="$run/train.jsonl"
  export COSMOS_AV_ACTION_NORMALIZER=""
  export BASE_CHECKPOINT_PATH="$init"
  export WAN_VAE_PATH="$WAN_VAE"
  echo "[EXPORT] checkpoint=$ckpt -> $model"
  python -m cosmos3eval.cli export --framework-root "$FRAMEWORK" \
    --dcp "$ckpt" --config "$cfg" --out "$model" --execute \
    2>&1 | tee "$run/export.log"
  [[ -d "$model" ]] || fatal "Export directory was not created: $model"
  printf '%s\n' "$ckpt" > "${model}.source_dcp.txt"
  sha256sum "$ckpt/model/.metadata" > "${model}.source_model_metadata.sha256"
}

export_models(){
  export_one "$S1_CKPT" "$S1_CONFIG" "$S1" "$MODEL1" "$EDGE_DCP"
  export_one "$S2_CKPT" "$S2_CONFIG" "$S2" "$MODEL2" "$S1_CKPT"
  echo "[EXPORT OK] $MODEL1 and $MODEL2"
}

evaluate(){
  [[ -d "$MODEL1" && -d "$MODEL2" ]] || fatal 'Both exported model directories are required'
  [[ ! -e "$EVAL/selection.jsonl" ]] || fatal "Evaluation selection already exists: $EVAL; use another RUN_TAG"
  [[ ! -e "$PRED1" && ! -e "$PRED2" ]] || fatal 'Prediction output already exists; use another RUN_TAG'
  python -m cosmos3eval.cli prepare --processed-root "$PROCESSED" \
    --datasets av2 --split val --max-clips-per-dataset "$EVAL_CLIPS" --seed "$SEED" --out "$EVAL"
  python - "$PROCESSED" "$EVAL" "$MATCH_CLIP_ID" "$SEED" <<'PYCODE'
import sys
from pathlib import Path
from cosmos3eval.data import read_jsonl, write_jsonl, make_request, verify_row
processed, ev, clip_id, seed = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
rows = read_jsonl(processed / 'index.jsonl')
matched = next((r for r in rows if r['clip_id'] == clip_id), None)
if matched is None:
    print(f'[EVAL] Reference clip not in processed index: {clip_id}')
elif matched['dataset'] != 'av2' or matched['split'] != 'val':
    print(f'[EVAL] Reference clip not in AV2 val split; not adding: {clip_id}')
else:
    selected = read_jsonl(ev / 'selection.jsonl')
    if not any(r['clip_id'] == clip_id for r in selected):
        verify_row(matched)
        selected.append(matched)
        write_jsonl(ev / 'selection.jsonl', selected)
        write_jsonl(ev / 'inference_requests.jsonl', [make_request(r, seed) for r in selected])
        print(f'[EVAL] Added reference AV2 validation clip: {clip_id}')
    else:
        print(f'[EVAL] Reference clip already included: {clip_id}')
PYCODE
  python -m cosmos3eval.cli score --eval-dir "$EVAL" --pred-dir "$EVAL/oracle_unused_pred" \
    --out "$EVAL/oracle_metrics" --oracle --model gt_oracle --no-plots
  python -m cosmos3eval.cli infer --eval-dir "$EVAL" --pred-dir "$PRED1" \
    --framework-root "$FRAMEWORK" --checkpoint "$MODEL1" --gpus 1 --execute
  python -m cosmos3eval.cli score --eval-dir "$EVAL" --pred-dir "$PRED1" \
    --model "$(basename "$S1")"
  python -m cosmos3eval.cli infer --eval-dir "$EVAL" --pred-dir "$PRED2" \
    --framework-root "$FRAMEWORK" --checkpoint "$MODEL2" --gpus 1 --execute
  python -m cosmos3eval.cli score --eval-dir "$EVAL" --pred-dir "$PRED2" \
    --model "$(basename "$S2")"
  python -m cosmos3eval.cli compare --summaries \
    "$PRED1/metrics/summary.json" "$PRED2/metrics/summary.json" \
    --out "$EVAL/stage1_vs_stage2.csv"
  echo "[EVALUATION OK] $EVAL/stage1_vs_stage2.csv"
}

print_config
case "$MODE" in
  stage1) stage1 ;;
  stage2) stage2 ;;
  export) export_models ;;
  evaluate) evaluate ;;
  all) stage1; stage2; export_models; evaluate ;;
  *) fatal 'Usage: bash retrain_cosmos3_av_repro.sh [stage1|stage2|export|evaluate|all]' ;;
esac
