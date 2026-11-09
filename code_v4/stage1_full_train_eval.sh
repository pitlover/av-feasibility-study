#!/usr/bin/env bash
# Cosmos3-Edge AV Stage1-only experiment: KITTI+Waymo train, held-out validation, and visualization.
# Requires the user's existing cosmos3av and cosmos3eval packages installed at the project root.
# Usage: bash stage1_full_train_eval.sh inspect|plan|train|export|prepare|oracle|infer|score|render|evaluate|all
# Safe defaults: raw actions, separate run names, no overwrite of checkpoints or predictions.
set -Eeuo pipefail

# Usage:
#   bash stage1_full_train_eval.sh --toml configs/stage1_full.toml plan
#   bash stage1_full_train_eval.sh train --toml configs/stage1_full.toml
#   bash stage1_full_train_eval.sh --toml configs/stage1_full.toml train.max_iters=500 plan
# Priority: dotted CLI assignment > TOML > environment variables > defaults.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_HELPER="$SCRIPT_DIR/scripts/stage1_toml_config.py"
[[ -f "$CONFIG_HELPER" ]] || { echo "[ERROR] Missing TOML helper: $CONFIG_HELPER" >&2; exit 2; }
MODE=""
CONFIG_PATH=""
OVERRIDES=()
WANDB_ENV_GIVEN="${WANDB_MODE+x}"
WANDB_OVERRIDE_GIVEN=0
while (( $# )); do
  case "$1" in
    --toml)
      (( $# >= 2 )) || { echo '[ERROR] --toml requires a file path' >&2; exit 2; }
      CONFIG_PATH="$2"; shift 2;;
    --toml=*) CONFIG_PATH="${1#--toml=}"; shift;;
    -h|--help)
      echo 'Usage: bash stage1_full_train_eval.sh [--toml FILE] inspect|config|plan|train|export|prepare|oracle|infer|score|render|evaluate|all [section.key=value ...]'
      exit 0;;
    inspect|config|plan|train|export|prepare|oracle|infer|score|render|evaluate|all)
      [[ -z "$MODE" ]] || { echo "[ERROR] Duplicate command: $1" >&2; exit 2; }
      MODE="$1"; shift;;
    *=*)
      [[ "$1" == wandb.* ]] && WANDB_OVERRIDE_GIVEN=1
      OVERRIDES+=("$1"); shift;;
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

# Paths in TOML may be relative to the project base.
BASE="$(cd "$BASE" && pwd)"
FRAMEWORK="${FRAMEWORK:-$BASE/cosmos-framework}"
PROCESSED="${PROCESSED:-$BASE/processed_virtual_native_v2}"
TEMPLATE_RUN="${TEMPLATE_RUN:-$BASE/runs_virtual/stage1_joint_300}"
for pathvar in FRAMEWORK PROCESSED TEMPLATE_RUN EDGE_DCP WAN_VAE EVAL_DIR_OVERRIDE PRED_DIR_OVERRIDE; do
  pth="${!pathvar}"
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

S1="$BASE/runs_stage1_full/$RUN_TAG"
S1_OUTPUT="$S1/outputs/cosmos3_av_id/pretrain/$RUN_TAG"
S1_DCP="$S1_OUTPUT/checkpoints/$(printf 'iter_%09d' "$STEP")"
S1_FINAL_DCP="$S1_OUTPUT/checkpoints/$(printf 'iter_%09d' "$MAX_ITERS")"
S1_CFG="$S1_OUTPUT/config.yaml"
MODEL="$BASE/exports/${RUN_TAG}_iter_${STEP}"
EVAL="$BASE/evaluation/stage1_${RUN_TAG}/kitti_waymo_${EVAL_TAG}"
PRED="$BASE/evaluation/pred_${RUN_TAG}_iter_${STEP}_${EVAL_TAG}"
# Optional explicit evaluation/prediction paths (existing results, e.g. *_retry1).
[[ -z "${EVAL_DIR_OVERRIDE:-}" ]] || EVAL="$EVAL_DIR_OVERRIDE"
[[ -z "${PRED_DIR_OVERRIDE:-}" ]] || PRED="$PRED_DIR_OVERRIDE"
export PYTHONPATH="$BASE:$FRAMEWORK${PYTHONPATH:+:$PYTHONPATH}"

fail(){ printf '[ERROR] %s\n' "$*" >&2; exit 1; }
need_file(){ [[ -f "$1" ]] || fail "Missing file: $1"; }
need_dcp(){ need_file "$1/model/.metadata"; need_file "$1/trainer/.metadata"; }
[[ "$RUN_TAG" =~ ^[a-zA-Z0-9_-]+$ && "$EVAL_TAG" =~ ^[a-zA-Z0-9_-]+$ ]] || fail 'RUN_TAG and EVAL_TAG may only contain letters, digits, _ and -'
for n in GPUS MAX_ITERS SAVE_EVERY EPOCH_CLIPS SEED EVAL_CLIPS STEP INFER_GPUS; do
  [[ "${!n}" =~ ^[0-9]+$ ]] || fail "$n must be a nonnegative integer"
done
(( GPUS>=1 && MAX_ITERS>=1 && SAVE_EVERY>=1 && EPOCH_CLIPS>=1 && EVAL_CLIPS>=1 && STEP>=1 && INFER_GPUS>=1 )) || fail 'Invalid nonpositive integer option'
(( MAX_ITERS % SAVE_EVERY == 0 )) || fail 'MAX_ITERS must be divisible by SAVE_EVERY for the requested final checkpoint check'

# Read-only evaluation must not depend on a historical Stage1 checkpoint or VAE.
case "$MODE" in
  inspect|plan|train|export|prepare|oracle|infer|all)
    need_file "$PROCESSED/index.jsonl" ;;
  evaluate)
    [[ "$EVAL_MODE" != full ]] || need_file "$PROCESSED/index.jsonl" ;;
esac
case "$MODE" in
  plan|train|export|all)
    if [[ -z "$EDGE_DCP" || -z "$WAN_VAE" ]]; then
      need_file "$TEMPLATE_RUN/run.json"
    fi
    EDGE_DCP="${EDGE_DCP:-$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["init_checkpoint"])' "$TEMPLATE_RUN/run.json")}"
    WAN_VAE="${WAN_VAE:-$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["wan_vae"])' "$TEMPLATE_RUN/run.json")}"
    [[ -d "$EDGE_DCP" ]] || fail "Cosmos3-Edge pretrained checkpoint missing: $EDGE_DCP"
    need_file "$WAN_VAE"
    need_file "$FRAMEWORK/cosmos_framework/scripts/train.py"
    ;;
  infer) need_file "$FRAMEWORK/cosmos_framework/scripts/train.py" ;;
  evaluate) [[ "$EVAL_MODE" != full ]] || need_file "$FRAMEWORK/cosmos_framework/scripts/train.py" ;;
esac

printf '[CONFIG] mode=%s run=%s iter=%s export_step=%s gpus=%s\n' "$MODE" "$RUN_TAG" "$MAX_ITERS" "$STEP" "$GPUS"
printf '[CONFIG] processed=%s eval=%s pred=%s eval_mode=%s\n' "$PROCESSED" "$EVAL" "$PRED" "$EVAL_MODE"
printf '[CONFIG] source_toml=%s weights=kitti:%s,waymo:%s wandb=%s\n' \
  "${CONFIG_PATH:-none}" "$KITTI_WEIGHT" "$WAYMO_WEIGHT" "$WANDB_MODE"


# Refuse accidental changes to the training plan between `plan` and `train`.
# Changes to export/eval settings (including export.step) are allowed.
check_frozen_config(){
  [[ -f "$S1/resolved_stage1_config.json" ]] || return 0  # historical runs
  local actual="$(mktemp)"
  local cmd=(python "$CONFIG_HELPER" --format json)
  [[ -z "$CONFIG_PATH" ]] || cmd+=(--toml "$CONFIG_PATH")
  if ! "${cmd[@]}" "${OVERRIDES[@]}" > "$actual"; then rm -f "$actual"; fail 'Cannot resolve TOML'; fi
  if ! python - "$S1/resolved_stage1_config.json" "$actual" <<'PYCOMPARE'
import json,sys
prev=json.load(open(sys.argv[1])); now=json.load(open(sys.argv[2]))
keys=('job.run_tag','paths.processed','data.epoch_clips','data.seed',
      'data.weights.kitti','data.weights.waymo','train.gpus',
      'train.max_iters','train.save_every','optimizer.lr')
diffs=[(k,prev.get(k),now.get(k)) for k in keys if prev.get(k)!=now.get(k)]
if diffs:
    for k,a,b in diffs: print(f'[CONFIG MISMATCH] {k}: planned={a!r} current={b!r}')
    sys.exit(1)
PYCOMPARE
  then rm -f "$actual"; fail 'Training settings differ from the original plan; use the same TOML or a new RUN_TAG'; fi
  rm -f "$actual"
}

inspect(){
  python - "$PROCESSED/index.jsonl" <<'PY'
import json, sys
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
p=Path(sys.argv[1]);rows=[json.loads(line) for line in p.read_text().splitlines() if line.strip()]
print('\n[DATASET INDEX]')
print('dataset   split    clips   sequences')
for ds in ('kitti','waymo','av2'):
    for split in ('train','val'):
        subset=[r for r in rows if r['dataset']==ds and r['split']==split]
        print(f'{ds:8s}  {split:5s}  {len(subset):6d}  {len({r["sequence"] for r in subset}):6d}')
print('\n[RAW TRANSLATION STATS: TRAIN ONLY, up to 200 unique clips/dataset]')
for ds in ('kitti','waymo'):
    subset=sorted((r for r in rows if r['dataset']==ds and r['split']=='train'), key=lambda x:x['clip_id'])
    if not subset: print('[MISSING]',ds,'train');continue
    rng=np.random.default_rng(42)
    sel=rng.choice(len(subset),size=min(200,len(subset)),replace=False)
    arr=np.concatenate([np.load(subset[int(i)]['action_file'],allow_pickle=False)[:,:3] for i in sel])
    print(f'{ds}: {len(sel)} clips ({len(arr)} actions), camera-frame translation [m/step]')
    for j,axis in enumerate('XYZ'):
        print(f'  {axis}: mean={np.mean(arr[:,j]):+.5f}, std={np.std(arr[:,j]):.5f}, q01={np.quantile(arr[:,j],.01):+.5f}, q99={np.quantile(arr[:,j],.99):+.5f}')
for ds in ('kitti','waymo'):
    tr={r['sequence'] for r in rows if r['dataset']==ds and r['split']=='train'}
    va={r['sequence'] for r in rows if r['dataset']==ds and r['split']=='val'}
    if not va:
        print(f'[WARNING] {ds} has NO validation clips; prepare cannot run until held-out data is preprocessed.')
    overlap=tr&va
    if overlap:print(f'[WARNING] {ds}: TRAIN/VAL sequence overlap: {sorted(overlap)[:10]}')
PY
}

plan(){
  [[ ! -e "$S1" ]] || fail "Existing run directory: $S1. Choose a new RUN_TAG (never overwrite)."
  echo "[PLAN] KITTI + Waymo, raw 9D actions; weights=kitti:$KITTI_WEIGHT,waymo:$WAYMO_WEIGHT"
  python -m cosmos3av.cli plan --stage pretrain --datasets kitti waymo \
    --weights "kitti:${KITTI_WEIGHT},waymo:${WAYMO_WEIGHT}" --processed-root "$PROCESSED" \
    --init-checkpoint "$EDGE_DCP" --wan-vae "$WAN_VAE" \
    --outdir "$S1" --epoch-clips "$EPOCH_CLIPS" \
    --max-iters "$MAX_ITERS" --save-every "$SAVE_EVERY" --lr "$LR" \
    --gpus "$GPUS" --seed "$SEED" --wandb-mode "${WANDB_MODE:-disabled}" --full-preflight
  # Store a copy of the declared and resolved experiment configuration.
  [[ -z "$CONFIG_PATH" ]] || cp "$CONFIG_PATH" "$S1/source_stage1_config.toml"
  json_cmd=(python "$CONFIG_HELPER" --format json)
  [[ -z "$CONFIG_PATH" ]] || json_cmd+=(--toml "$CONFIG_PATH")
  "${json_cmd[@]}" "${OVERRIDES[@]}" > "$S1/resolved_stage1_config.json"
  python -m cosmos3av.cli smoke --manifest "$S1/train.jsonl"
  python - "$S1/run.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r['stage']=='pretrain' and r['action_frame']=='camera'
assert r['action_normalizer'] is None, 'This baseline must remain RAW action'
print('[PLAN OK] raw camera actions; train entries:',r['n_train'],'val entries:',r['n_val'])
PY
}

train(){
  need_file "$S1/run.json"
  check_frozen_config
  [[ ! -e "$S1/train.log" ]] || fail "Training log already exists: $S1/train.log (refusing to relaunch implicitly)"
  [[ ! -e "$S1_FINAL_DCP" ]] || fail "Final DCP already exists: $S1_FINAL_DCP"
  unset IMAGINAIRE_OUTPUT_ROOT || true
  echo '[TRAIN] Starting training. Logs and output checkpoints remain under the new run directory.'
  # Without a new TOML/env override, preserve the mode saved by `plan`.
  local extra=()
  if [[ -n "$CONFIG_PATH" || -n "$WANDB_ENV_GIVEN" || "$WANDB_OVERRIDE_GIVEN" == 1 ]]; then
    extra=(--wandb-mode "$WANDB_MODE")
  fi
  python -m cosmos3av.cli launch --run-dir "$S1" --framework-root "$FRAMEWORK" \
    "${extra[@]}" --execute 2>&1 | tee "$S1/train.log"
  need_dcp "$S1_FINAL_DCP"
  echo "[TRAIN OK] $S1_FINAL_DCP"
}

export_model(){
  need_dcp "$S1_DCP"
  need_file "$S1_CFG"
  [[ ! -e "$MODEL" ]] || fail "Export already exists: $MODEL"
  [[ ! -e "${MODEL}.source_dcp.txt" ]] || fail "Export provenance exists: ${MODEL}.source_dcp.txt"
  export COSMOS_AV_MANIFEST="$S1/train.jsonl"
  export COSMOS_AV_ACTION_NORMALIZER=""
  export BASE_CHECKPOINT_PATH="$EDGE_DCP"
  export WAN_VAE_PATH="$WAN_VAE"
  echo "[EXPORT] $S1_DCP -> $MODEL"
  python -m cosmos3eval.cli export --framework-root "$FRAMEWORK" \
     --dcp "$S1_DCP" --config "$S1_CFG" --out "$MODEL" --execute 2>&1 | tee "$S1/export_iter_${STEP}.log"
  [[ -d "$MODEL" ]] || fail "Export directory missing: $MODEL"
  printf '%s\n' "$S1_DCP" > "${MODEL}.source_dcp.txt"
  sha256sum "$S1_DCP/model/.metadata" > "${MODEL}.source_model_metadata.sha256"
  echo "[EXPORT OK] $MODEL"
}

prepare_eval(){
  if [[ -f "$EVAL/selection.jsonl" && -f "$EVAL/inference_requests.jsonl" ]]; then
    echo "[PREPARE] Reusing existing fixed evaluation selection: $EVAL"
    return
  fi
  [[ ! -e "$EVAL" ]] || fail "Incomplete evaluation dir: $EVAL; choose another EVAL_TAG"
  python - "$PROCESSED" "$EVAL" "$EVAL_CLIPS" "$SEED" <<'PY'
import sys, json, random
from pathlib import Path
from collections import defaultdict
from cosmos3eval.data import read_jsonl, write_jsonl, verify_row, make_request
root,out=map(Path,sys.argv[1:3]);num=int(sys.argv[3]);seed=int(sys.argv[4])
rows=read_jsonl(root/'index.jsonl')
selected=[]
for ds in ('kitti','waymo'):
    train={r['sequence'] for r in rows if r['dataset']==ds and r['split']=='train'}
    val=[r for r in rows if r['dataset']==ds and r['split']=='val']
    if not val:raise SystemExit(f'[ERROR] No {ds}/val clips in index; preprocess held-out {ds} validation first')
    overlap=train&{r['sequence'] for r in val}
    if overlap:raise SystemExit(f'[ERROR] {ds} train/val sequence overlap: {sorted(overlap)[:5]}')
    groups=defaultdict(list)
    for r in val:groups[r['sequence']].append(r)
    rng=random.Random(seed + (1 if ds=='waymo' else 0))
    keys=sorted(groups)
    rng.shuffle(keys)
    for key in keys:rng.shuffle(groups[key])
    chosen=[]
    while len(chosen)<min(num,len(val)):
        found=False
        for key in keys:
            if groups[key]:
                chosen.append(groups[key].pop());found=True
                if len(chosen)>=min(num,len(val)):break
        if not found:break
    for row in chosen:verify_row(row)
    selected.extend(chosen)
    print(f'[PREPARE] {ds}: {len(chosen)} clips from {len({r["sequence"] for r in chosen})} val sequences')
out.mkdir(parents=True,exist_ok=False)
write_jsonl(out/'selection.jsonl',selected)
write_jsonl(out/'inference_requests.jsonl',[make_request(r,seed) for r in selected])
print('[PREPARE OK]',out,'total clips=',len(selected))
PY
}

oracle(){
  need_file "$EVAL/selection.jsonl"
  if [[ -f "$EVAL/oracle_metrics/summary.json" ]]; then
    echo "[ORACLE] Reusing existing GT-only validation check: $EVAL/oracle_metrics/summary.json"
    return
  fi
  python -m cosmos3eval.cli score --eval-dir "$EVAL" --pred-dir "$EVAL/unused_oracle_predictions" \
    --out "$EVAL/oracle_metrics" --model gt_oracle --oracle --no-plots
  echo '[ORACLE OK] Ground-truth action decoding is internally consistent.'
}

infer(){
  need_file "$EVAL/selection.jsonl"
  need_file "$EVAL/inference_requests.jsonl"
  [[ -d "$MODEL" ]] || fail "Exported model missing: $MODEL; run export with STEP=$STEP"
  [[ ! -e "$PRED" ]] || fail "Prediction directory already exists: $PRED; use new EVAL_TAG or inspect previous results"
  python -m cosmos3eval.cli infer --eval-dir "$EVAL" --pred-dir "$PRED" \
     --framework-root "$FRAMEWORK" --checkpoint "$MODEL" --gpus "$INFER_GPUS" --seed "$SEED" --execute
  [[ -n "$(find "$PRED" -mindepth 2 -maxdepth 2 -name 'sample_outputs.json' -print -quit 2>/dev/null)" ]] || fail "No sample_outputs.json found in $PRED"
  printf '%s\n' "$MODEL" > "${PRED}.source_export.txt"
  echo "[INFER OK] $PRED"
}

# Verify the complete evaluation selection before scoring or rendering. A partially
# created predictions directory must not silently appear to be a complete run.
verify_predictions(){
  need_file "$EVAL/selection.jsonl"
  [[ -d "$PRED" ]] || fail "Missing predictions directory: $PRED (set eval.pred_dir in TOML)"
  python - "$EVAL/selection.jsonl" "$PRED" <<'PY_PRED_CHECK'
import sys
from pathlib import Path
from cosmos3eval.data import read_jsonl
from cosmos3eval.predictions import load_pred
sel, root = Path(sys.argv[1]), Path(sys.argv[2])
rows = read_jsonl(sel)
missing = []
for row in rows:
    try:
        load_pred(root, row['clip_id'])
    except (FileNotFoundError, ValueError, KeyError, TypeError) as exc:
        missing.append((row['clip_id'], str(exc)))
print(f'[PRED CHECK] selected={len(rows)} available={len(rows)-len(missing)} missing={len(missing)}')
if missing:
    for cid,why in missing[:10]:
        print(f'[PRED MISSING] {cid}: {why}', file=sys.stderr)
    raise SystemExit('[ERROR] Missing or unreadable predictions; run infer to a NEW eval tag or set eval.pred_dir')
PY_PRED_CHECK
}

score(){
  verify_predictions
  local mode="$WANDB_MODE"
  python -m cosmos3eval.cli score --eval-dir "$EVAL" --pred-dir "$PRED" \
    --model "${RUN_TAG}_iter_${STEP}" \
    --wandb-mode "$mode" --wandb-project "$WANDB_PROJECT" --wandb-group pretrain \
    --wandb-name "eval_${RUN_TAG}_iter_${STEP}_${EVAL_TAG}"
  python - "$PRED/metrics/summary.json" <<'PY'
import json,sys
p=sys.argv[1]; summary=json.load(open(p)); d=summary['overall']
print('\n[STAGE1 VALIDATION OVERALL]')
for key in ('model','n_clips','n_failed','rre_deg_mean','rte_m_mean','ate_m_rmse','final_position_error_m_mean'):
    print(f'{key:30s}: {d.get(key)}')
print('\n[BY DATASET]')
for ds, s in summary['by_dataset'].items():
    print(ds, 'clips=',s['n_clips'],'RRE=',round(s['rre_deg_mean'],4),'RTE=',round(s['rte_m_mean'],4),'ATE_RMSE=',round(s['ate_m_rmse'],4))
print('\nFull results:',p)
PY
}

render(){
  verify_predictions
  local selected ds cid
  selected="$(python "$SCRIPT_DIR/scripts/choose_render_clips.py" \
    --selection "$EVAL/selection.jsonl" --datasets kitti waymo \
    --max-per-dataset "$RENDER_MAX_CLIPS" --clip-ids-json "$RENDER_CLIP_IDS_JSON")" \
    || fail "Could not select Stage1 clips for rendering"
  while IFS=$'\t' read -r ds cid; do
    [[ -n "$cid" ]] || continue
    echo "[RENDER] $ds clip=$cid (ground_height_m=$GROUND_HEIGHT_M approximate)"
    python -m cosmos3eval.cli render --eval-dir "$EVAL" --pred-dir "$PRED" \
      --clip-id "$cid" --layout "$RENDER_LAYOUT" --horizon-steps "$HORIZON_STEPS" \
      --ground-height-m "$GROUND_HEIGHT_M" --fps "$RENDER_FPS" \
      --out-dir "$PRED/qualitative/$ds"
  done <<< "$selected"
  echo "[RENDER OK] $PRED/qualitative/"
}
maybe_render(){
  if [[ "$RENDER_ENABLED" == true ]]; then render
  else echo '[RENDER SKIPPED] eval.render_enabled=false (run the render command to render separately)'; fi
}

evaluate_from_toml(){
  case "$EVAL_MODE" in
    score) score ;;
    render) render ;;
    both) score; render ;;
    full) prepare_eval; oracle; infer; score; maybe_render ;;
    *) fail "Unsupported eval.mode: $EVAL_MODE" ;;
  esac
}

case "$MODE" in
  inspect) inspect;;
  plan) plan;;
  train) train;;
  export) export_model;;
  prepare) prepare_eval;;
  oracle) prepare_eval; oracle;;
  infer) infer;;
  score) score;;
  render) render;;
  evaluate) evaluate_from_toml;;
  all) [[ "$EVAL_MODE" == full ]] || fail 'all requires eval.mode="full"'; inspect; plan; train; export_model; evaluate_from_toml;;
  *) fail 'Usage: bash stage1_full_train_eval.sh [--toml FILE] inspect|config|plan|train|export|prepare|oracle|infer|score|render|evaluate|all [section.key=value]';;
esac
