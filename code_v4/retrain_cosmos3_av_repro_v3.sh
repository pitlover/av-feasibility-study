#!/usr/bin/env bash
# Reproducible Cosmos3-Edge AV-ID experiment with Stage2 TOML + W&B support.
# Usage examples:
#   bash retrain_cosmos3_av_repro_v3.sh --toml configs/stage2_repro.toml config
#   bash retrain_cosmos3_av_repro_v3.sh --toml configs/stage2_repro.toml stage1
#   bash retrain_cosmos3_av_repro_v3.sh --toml configs/stage2_repro.toml stage2
#   bash retrain_cosmos3_av_repro_v3.sh --toml configs/stage2_repro.toml export
#   bash retrain_cosmos3_av_repro_v3.sh --toml configs/stage2_repro.toml evaluate
#   bash retrain_cosmos3_av_repro_v3.sh --toml configs/stage2_repro.toml all
# Priority: dotted CLI assignment > TOML > environment variables > defaults.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_HELPER="$SCRIPT_DIR/scripts/stage2_toml_config.py"
[[ -f "$CONFIG_HELPER" ]] || { echo "[ERROR] Missing TOML helper: $CONFIG_HELPER" >&2; exit 2; }
MODE=""
CONFIG_PATH=""
OVERRIDES=()
while (( $# )); do
  case "$1" in
    --toml)
      (( $# >= 2 )) || { echo '[ERROR] --toml requires a file path' >&2; exit 2; }
      CONFIG_PATH="$2"; shift 2;;
    --toml=*) CONFIG_PATH="${1#--toml=}"; shift;;
    -h|--help)
      echo 'Usage: bash retrain_cosmos3_av_repro_v3.sh [--toml FILE] config|stage1|stage2|export|evaluate|all [section.key=value ...]'
      exit 0;;
    config|stage1|stage2|export|evaluate|all)
      [[ -z "$MODE" ]] || { echo "[ERROR] Duplicate command: $1" >&2; exit 2; }
      MODE="$1"; shift;;
    *=*) OVERRIDES+=("$1"); shift;;
    *) echo "[ERROR] Unknown argument: $1" >&2; exit 2;;
  esac
done
MODE="${MODE:-config}"
if [[ -n "$CONFIG_PATH" ]]; then
  CONFIG_PATH="$(python -c 'from pathlib import Path;import sys;print(Path(sys.argv[1]).expanduser().resolve())' "$CONFIG_PATH")"
fi

CFG_OUT="$(mktemp)"
trap 'rm -f "$CFG_OUT"' EXIT
CFG_CMD=(python "$CONFIG_HELPER" --format env0)
[[ -z "$CONFIG_PATH" ]] || CFG_CMD+=(--toml "$CONFIG_PATH")
CFG_CMD+=("${OVERRIDES[@]}")
"${CFG_CMD[@]}" > "$CFG_OUT" || exit 2
while IFS= read -r -d '' key && IFS= read -r -d '' value; do
  printf -v "$key" '%s' "$value"
done < "$CFG_OUT"
rm -f "$CFG_OUT"

BASE="$(cd "$BASE" && pwd)"
FRAMEWORK="${FRAMEWORK:-$BASE/cosmos-framework}"
PROCESSED="${PROCESSED:-$BASE/processed_virtual_native_v2}"
TEMPLATE_RUN="${TEMPLATE_RUN:-$BASE/runs_virtual/stage1_joint_300}"
for pathvar in FRAMEWORK PROCESSED TEMPLATE_RUN EDGE_DCP WAN_VAE; do
  pth="${!pathvar:-}"
  if [[ -n "$pth" && "$pth" != /* ]]; then
    printf -v "$pathvar" '%s' "$BASE/$pth"
  fi
done

if [[ "$MODE" == config ]]; then
  echo "[TOML] ${CONFIG_PATH:-<default/environment>}"
  cfg_cmd=(python "$CONFIG_HELPER" --format json)
  [[ -z "$CONFIG_PATH" ]] || cfg_cmd+=(--toml "$CONFIG_PATH")
  "${cfg_cmd[@]}" "${OVERRIDES[@]}"
  exit 0
fi

S1="$BASE/runs_repro/stage1_joint_${MAX_ITERS}_${RUN_TAG}"
S2="$BASE/runs_repro/stage2_av2_${MAX_ITERS}_${RUN_TAG}"
S1_CKPT="$S1/outputs/cosmos3_av_id/pretrain/$(basename "$S1")/checkpoints/$(printf 'iter_%09d' "$MAX_ITERS")"
S2_CKPT="$S2/outputs/cosmos3_av_id/posttrain/$(basename "$S2")/checkpoints/$(printf 'iter_%09d' "$MAX_ITERS")"
S1_CONFIG="$S1/outputs/cosmos3_av_id/pretrain/$(basename "$S1")/config.yaml"
S2_CONFIG="$S2/outputs/cosmos3_av_id/posttrain/$(basename "$S2")/config.yaml"
MODEL1="$BASE/exports/$(basename "$S1")"
MODEL2="$BASE/exports/$(basename "$S2")"
EVAL="$BASE/evaluation/av2_val_${EVAL_TAG}"
PRED1="$BASE/evaluation/pred_$(basename "$S1")_${EVAL_TAG}"
PRED2="$BASE/evaluation/pred_$(basename "$S2")_${EVAL_TAG}"
export PYTHONPATH="$BASE:$FRAMEWORK${PYTHONPATH:+:$PYTHONPATH}"

fatal(){ echo "[ERROR] $*" >&2; exit 1; }
require_file(){ [[ -f "$1" ]] || fatal "Missing file: $1"; }
require_ckpt(){ require_file "$1/model/.metadata"; require_file "$1/trainer/.metadata"; }

require_file "$PROCESSED/index.jsonl"
require_file "$FRAMEWORK/cosmos_framework/scripts/train.py"
if [[ -z "${EDGE_DCP:-}" || -z "${WAN_VAE:-}" ]]; then
  require_file "$TEMPLATE_RUN/run.json"
fi
EDGE_DCP="${EDGE_DCP:-$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["init_checkpoint"])' "$TEMPLATE_RUN/run.json")}"
WAN_VAE="${WAN_VAE:-$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["wan_vae"])' "$TEMPLATE_RUN/run.json")}"
[[ -d "$EDGE_DCP" ]] || fatal "Cosmos3-Edge checkpoint missing: $EDGE_DCP"
require_file "$WAN_VAE"

[[ "$RUN_TAG" =~ ^[a-zA-Z0-9_-]+$ && "$EVAL_TAG" =~ ^[a-zA-Z0-9_-]+$ ]] || fatal 'RUN_TAG and EVAL_TAG may only contain letters, digits, _ and -'
for n in GPUS MAX_ITERS SAVE_EVERY EPOCH_CLIPS EVAL_CLIPS SEED INFER_GPUS; do
  [[ "${!n}" =~ ^[0-9]+$ ]] || fatal "$n must be a nonnegative integer"
done
(( GPUS>=1 && MAX_ITERS>=1 && SAVE_EVERY>=1 && EPOCH_CLIPS>=1 && EVAL_CLIPS>=1 && INFER_GPUS>=1 )) || fatal 'Invalid nonpositive integer option'
(( MAX_ITERS % SAVE_EVERY == 0 )) || fatal 'MAX_ITERS must be divisible by SAVE_EVERY'
[[ "$WANDB_MODE" =~ ^(disabled|online|offline)$ ]] || fatal 'WANDB_MODE must be disabled, online, or offline'

print_config(){
  printf '\n[CONFIG] mode=%s run_tag=%s eval_tag=%s gpus=%s iters=%s wandb=%s\n' "$MODE" "$RUN_TAG" "$EVAL_TAG" "$GPUS" "$MAX_ITERS" "$WANDB_MODE"
  printf '[CONFIG] BASE=%s\n[CONFIG] PROCESSED=%s\n' "$BASE" "$PROCESSED"
  printf '[CONFIG] EDGE_DCP=%s\n[CONFIG] WAN_VAE=%s\n' "$EDGE_DCP" "$WAN_VAE"
  printf '[CONFIG] Stage1=%s\n[CONFIG] Stage2=%s\n' "$S1" "$S2"
  printf '[CONFIG] EVAL=%s\n[CONFIG] weights=kitti:%s,waymo:%s\n' "$EVAL" "$KITTI_WEIGHT" "$WAYMO_WEIGHT"
}

stage1(){
  [[ ! -e "$S1" ]] || fatal "Stage1 run already exists; use a new RUN_TAG: $S1"
  echo '[STAGE1] Creating KITTI+Waymo pretraining plan'
  python -m cosmos3av.cli plan \
    --stage pretrain --datasets kitti waymo --weights "kitti:${KITTI_WEIGHT},waymo:${WAYMO_WEIGHT}" \
    --processed-root "$PROCESSED" --init-checkpoint "$EDGE_DCP" --wan-vae "$WAN_VAE" \
    --outdir "$S1" --epoch-clips "$EPOCH_CLIPS" --max-iters "$MAX_ITERS" \
    --save-every "$SAVE_EVERY" --gpus "$GPUS" --seed "$SEED" \
    --wandb-mode "$WANDB_MODE" --full-preflight
  [[ -z "$CONFIG_PATH" ]] || cp "$CONFIG_PATH" "$S1/source_stage2_config.toml"
  json_cmd=(python "$CONFIG_HELPER" --format json)
  [[ -z "$CONFIG_PATH" ]] || json_cmd+=(--toml "$CONFIG_PATH")
  "${json_cmd[@]}" "${OVERRIDES[@]}" > "$S1/resolved_stage2_config.json"
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
    --save-every "$SAVE_EVERY" --gpus "$GPUS" --seed "$SEED" \
    --wandb-mode "$WANDB_MODE" --full-preflight
  [[ -z "$CONFIG_PATH" ]] || cp "$CONFIG_PATH" "$S2/source_stage2_config.toml"
  json_cmd=(python "$CONFIG_HELPER" --format json)
  [[ -z "$CONFIG_PATH" ]] || json_cmd+=(--toml "$CONFIG_PATH")
  "${json_cmd[@]}" "${OVERRIDES[@]}" > "$S2/resolved_stage2_config.json"
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
  [[ ! -e "$EVAL/selection.jsonl" ]] || fatal "Evaluation selection already exists: $EVAL; use another EVAL_TAG"
  [[ ! -e "$PRED1" && ! -e "$PRED2" ]] || fatal 'Prediction output already exists; use another EVAL_TAG'
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
    --framework-root "$FRAMEWORK" --checkpoint "$MODEL1" --gpus "$INFER_GPUS" --execute
  python -m cosmos3eval.cli score --eval-dir "$EVAL" --pred-dir "$PRED1" \
    --model "$(basename "$S1")" --wandb-mode "$WANDB_MODE" --wandb-project "$WANDB_PROJECT" \
    --wandb-group pretrain --wandb-name "eval_$(basename "$S1")_${EVAL_TAG}"
  python -m cosmos3eval.cli infer --eval-dir "$EVAL" --pred-dir "$PRED2" \
    --framework-root "$FRAMEWORK" --checkpoint "$MODEL2" --gpus "$INFER_GPUS" --execute
  python -m cosmos3eval.cli score --eval-dir "$EVAL" --pred-dir "$PRED2" \
    --model "$(basename "$S2")" --wandb-mode "$WANDB_MODE" --wandb-project "$WANDB_PROJECT" \
    --wandb-group posttrain --wandb-name "eval_$(basename "$S2")_${EVAL_TAG}"
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
  *) fatal 'Usage: bash retrain_cosmos3_av_repro_v3.sh [config|stage1|stage2|export|evaluate|all]' ;;
esac
