#!/usr/bin/env bash
# Cosmos3-Edge AV Stage1-only experiment: KITTI+Waymo train, held-out validation, and visualization.
# Requires the user's existing cosmos3av and cosmos3eval packages installed at the project root.
# Usage: bash stage1_full_train_eval.sh inspect|plan|train|export|prepare|oracle|infer|score|render|evaluate|all
# Safe defaults: raw actions, separate run names, no overwrite of checkpoints or predictions.
set -Eeuo pipefail

MODE="${1:-inspect}"
BASE="${BASE:-$(pwd)}"; BASE="$(cd "$BASE" && pwd)"
FRAMEWORK="${FRAMEWORK:-$BASE/cosmos-framework}"
PROCESSED="${PROCESSED:-$BASE/processed_virtual_native_v2}"
TEMPLATE_RUN="${TEMPLATE_RUN:-$BASE/runs_virtual/stage1_joint_300}"
RUN_TAG="${RUN_TAG:-s1_full_raw_v1}"
EVAL_TAG="${EVAL_TAG:-val_seed42_v1}"
GPUS="${GPUS:-4}"
MAX_ITERS="${MAX_ITERS:-3000}"
SAVE_EVERY="${SAVE_EVERY:-500}"
EPOCH_CLIPS="${EPOCH_CLIPS:-12000}"
LR="${LR:-1e-5}"
SEED="${SEED:-42}"
EVAL_CLIPS="${EVAL_CLIPS:-20}"
STEP="${STEP:-$MAX_ITERS}"
INFER_GPUS="${INFER_GPUS:-1}"
GROUND_HEIGHT_M="${GROUND_HEIGHT_M:-1.7}"

S1="$BASE/runs_stage1_full/$RUN_TAG"
S1_OUTPUT="$S1/outputs/cosmos3_av_id/pretrain/$RUN_TAG"
S1_DCP="$S1_OUTPUT/checkpoints/$(printf 'iter_%09d' "$STEP")"
S1_FINAL_DCP="$S1_OUTPUT/checkpoints/$(printf 'iter_%09d' "$MAX_ITERS")"
S1_CFG="$S1_OUTPUT/config.yaml"
MODEL="$BASE/exports/${RUN_TAG}_iter_${STEP}"
EVAL="$BASE/evaluation/stage1_${RUN_TAG}/kitti_waymo_${EVAL_TAG}"
PRED="$BASE/evaluation/pred_${RUN_TAG}_iter_${STEP}_${EVAL_TAG}"
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

need_file "$PROCESSED/index.jsonl"
need_file "$TEMPLATE_RUN/run.json"
need_file "$FRAMEWORK/cosmos_framework/scripts/train.py"
EDGE_DCP="$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["init_checkpoint"])' "$TEMPLATE_RUN/run.json")"
WAN_VAE="$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["wan_vae"])' "$TEMPLATE_RUN/run.json")"
[[ -d "$EDGE_DCP" ]] || fail "Cosmos3-Edge pretrained checkpoint missing: $EDGE_DCP"
need_file "$WAN_VAE"

printf '[CONFIG] mode=%s run=%s iter=%s export_step=%s gpus=%s\n' "$MODE" "$RUN_TAG" "$MAX_ITERS" "$STEP" "$GPUS"
printf '[CONFIG] processed=%s eval=%s\n' "$PROCESSED" "$EVAL"

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
  echo '[PLAN] KITTI + Waymo, raw 9D actions; sampled manifest has 1:1 dataset weights'
  python -m cosmos3av.cli plan --stage pretrain --datasets kitti waymo \
    --weights kitti:1,waymo:1 --processed-root "$PROCESSED" \
    --init-checkpoint "$EDGE_DCP" --wan-vae "$WAN_VAE" \
    --outdir "$S1" --epoch-clips "$EPOCH_CLIPS" \
    --max-iters "$MAX_ITERS" --save-every "$SAVE_EVERY" --lr "$LR" \
    --gpus "$GPUS" --seed "$SEED" --full-preflight
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
  [[ ! -e "$S1/train.log" ]] || fail "Training log already exists: $S1/train.log (refusing to relaunch implicitly)"
  [[ ! -e "$S1_FINAL_DCP" ]] || fail "Final DCP already exists: $S1_FINAL_DCP"
  unset IMAGINAIRE_OUTPUT_ROOT || true
  echo '[TRAIN] Starting training. Logs and output checkpoints remain under the new run directory.'
  python -m cosmos3av.cli launch --run-dir "$S1" --framework-root "$FRAMEWORK" --execute 2>&1 | tee "$S1/train.log"
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

score(){
  need_file "$EVAL/selection.jsonl"
  [[ -d "$PRED" ]] || fail "Missing predictions: $PRED"
  python -m cosmos3eval.cli score --eval-dir "$EVAL" --pred-dir "$PRED" --model "${RUN_TAG}_iter_${STEP}" 
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
  need_file "$EVAL/selection.jsonl"
  [[ -d "$PRED" ]] || fail "Missing predictions: $PRED"
  for ds in kitti waymo; do
    cid="$(python - "$EVAL/selection.jsonl" "$ds" <<'PY'
import json,sys
for line in open(sys.argv[1]):
    row=json.loads(line)
    if row['dataset']==sys.argv[2]:print(row['clip_id']);break
PY
)"
    [[ -n "$cid" ]] || fail "No selected $ds clip"
    echo "[RENDER] $ds clip=$cid, ground_height_m=$GROUND_HEIGHT_M (approximate)"
    python -m cosmos3eval.cli render --eval-dir "$EVAL" --pred-dir "$PRED" \
      --clip-id "$cid" --layout both --horizon-steps 30 \
      --ground-height-m "$GROUND_HEIGHT_M" --out-dir "$PRED/qualitative/$ds"
  done
  echo "[RENDER OK] $PRED/qualitative/{kitti,waymo}/"
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
  evaluate) prepare_eval; oracle; infer; score; render;;
  all) inspect; plan; train; export_model; prepare_eval; oracle; infer; score; render;;
  *) fail 'Usage: bash stage1_full_train_eval.sh inspect|plan|train|export|prepare|oracle|infer|score|render|evaluate|all';;
esac
