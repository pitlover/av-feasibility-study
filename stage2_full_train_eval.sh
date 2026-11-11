#!/usr/bin/env bash
# Cosmos3-Edge AV Stage2 ONLY: use a completed Stage1 DCP -> AV2 post-train -> eval.
# Requires the previously applied cosmos3av/cosmos3eval W&B integration patch.
# Example:
#   bash stage2_full_train_eval.sh --toml configs/stage2_full.toml config
#   bash stage2_full_train_eval.sh --toml configs/stage2_full.toml plan
#   bash stage2_full_train_eval.sh --toml configs/stage2_full.toml train
#   bash stage2_full_train_eval.sh --toml configs/stage2_full.toml export
#   bash stage2_full_train_eval.sh --toml configs/stage2_full.toml evaluate
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELPER="$SCRIPT_DIR/scripts/stage2_toml_config.py"
[[ -f "$HELPER" ]] || { echo "[ERROR] TOML helper not found: $HELPER" >&2; exit 2; }
MODE=""
CONFIG_PATH=""
OVERRIDES=()
while (( $# )); do
  case "$1" in
    --toml)
      (( $# >= 2 )) || { echo '[ERROR] --toml needs FILE' >&2; exit 2; }
      CONFIG_PATH="$2"; shift 2;;
    --toml=*) CONFIG_PATH="${1#--toml=}"; shift;;
    config|inspect|plan|train|export|prepare|oracle|infer|score|render|evaluate|all)
      [[ -z "$MODE" ]] || { echo "[ERROR] Duplicate command: $1" >&2; exit 2; }
      MODE="$1"; shift;;
    -h|--help)
      echo 'Usage: bash stage2_full_train_eval.sh [--toml FILE] config|inspect|plan|train|export|prepare|oracle|infer|score|render|evaluate|all [section.key=value ...]'
      exit 0;;
    *=*) OVERRIDES+=("$1"); shift;;
    *) echo "[ERROR] Unknown argument: $1" >&2; exit 2;;
  esac
done
MODE="${MODE:-config}"
if [[ -n "$CONFIG_PATH" ]]; then
  CONFIG_PATH="$(python -c 'from pathlib import Path;import sys;print(Path(sys.argv[1]).expanduser().resolve())' "$CONFIG_PATH")"
fi
CONFIG_CMD=(python "$HELPER")
[[ -z "$CONFIG_PATH" ]] || CONFIG_CMD+=(--toml "$CONFIG_PATH")
CFG_OUT="$(mktemp)"
trap 'rm -f "$CFG_OUT"' EXIT
"${CONFIG_CMD[@]}" --format env0 "${OVERRIDES[@]}" > "$CFG_OUT" || exit 2
while IFS= read -r -d '' key && IFS= read -r -d '' value; do
  printf -v "$key" '%s' "$value"
done < "$CFG_OUT"
rm -f "$CFG_OUT"

BASE="$(cd "$BASE" && pwd)"
for var in FRAMEWORK PROCESSED STAGE1_RUN STAGE1_CKPT WAN_VAE EVAL_DIR_OVERRIDE PRED_DIR_OVERRIDE; do
  val="${!var}"
  if [[ -n "$val" && "$val" != /* ]]; then
    printf -v "$var" '%s' "$BASE/$val"
  fi
done
if [[ "$MODE" == config ]]; then
  echo '[CONFIG] Parsed TOML values (relative paths resolve under BASE when running):'
  "${CONFIG_CMD[@]}" --format json "${OVERRIDES[@]}"
  echo "[CONFIG] resolved Stage1 DCP: $STAGE1_CKPT"
  exit 0
fi

S2="$BASE/runs_stage2/$RUN_TAG"
S2_OUTPUT="$S2/outputs/cosmos3_av_id/posttrain/$RUN_TAG"
S2_FINAL_DCP="$S2_OUTPUT/checkpoints/$(printf 'iter_%09d' "$MAX_ITERS")"
S2_DCP="$S2_OUTPUT/checkpoints/$(printf 'iter_%09d' "$STEP")"
S2_CFG="$S2_OUTPUT/config.yaml"
MODEL="$BASE/exports/${RUN_TAG}_iter_${STEP}"
EVAL="$BASE/evaluation/stage2_${RUN_TAG}/av2_${EVAL_TAG}"
PRED="$BASE/evaluation/pred_${RUN_TAG}_iter_${STEP}_${EVAL_TAG}"
# Allow score/render to read predictions created by a different evaluation/infer run.
# In particular this supports manually generated *_retry1 prediction folders.
[[ -z "${EVAL_DIR_OVERRIDE:-}" ]] || EVAL="$EVAL_DIR_OVERRIDE"
[[ -z "${PRED_DIR_OVERRIDE:-}" ]] || PRED="$PRED_DIR_OVERRIDE"
export PYTHONPATH="$BASE:$FRAMEWORK${PYTHONPATH:+:$PYTHONPATH}"

fail(){ echo "[ERROR] $*" >&2; exit 1; }
need_file(){ [[ -f "$1" ]] || fail "Missing file: $1"; }
need_dcp(){ need_file "$1/model/.metadata"; need_file "$1/trainer/.metadata"; }

# Ensure Stage1 run lineage is explicit and valid; never use original Edge weights in Stage2.
validate_stage1(){
  need_file "$STAGE1_RUN/run.json"
  need_dcp "$STAGE1_CKPT"
  python - "$STAGE1_RUN" "$STAGE1_CKPT" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1]).resolve()
ckpt = Path(sys.argv[2]).resolve()
info = json.loads((run/'run.json').read_text(encoding='utf-8'))
if info.get('stage') != 'pretrain':
    raise SystemExit('[ERROR] stage1_run/run.json does not describe a Stage1 pretrain run')
if info.get('action_frame') != 'camera':
    raise SystemExit('[ERROR] Stage1 is not camera-relative; incompatible with current AV2 Stage2')
if info.get('action_normalizer') is not None:
    raise SystemExit('[ERROR] This script targets the existing raw-9D Stage1. Normalized Stage1 needs explicitly matched export/inference settings.')
if run / 'outputs' not in ckpt.parents:
    raise SystemExit('[ERROR] Stage1 checkpoint is not under stage1_run/outputs')
print('[STAGE1 CHECK] pretrain camera-relative raw 9D')
print('[STAGE1 CHECK] source checkpoint:', ckpt)
PY
}

# Read-only score/render must work even if the original Stage1 run has moved.
# Planning/training/exporting Stage2 still strictly requires the real Stage1 DCP.
case "$MODE" in
  plan|train|export|all)
    validate_stage1
    need_file "$FRAMEWORK/cosmos_framework/scripts/train.py"
    if [[ -z "$WAN_VAE" ]]; then
      WAN_VAE="$(python - "$STAGE1_RUN/run.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['wan_vae'])
PY
)"
    fi
    need_file "$WAN_VAE"
    ;;
  infer) need_file "$FRAMEWORK/cosmos_framework/scripts/train.py" ;;
  evaluate) [[ "$EVAL_MODE" != full ]] || need_file "$FRAMEWORK/cosmos_framework/scripts/train.py" ;;
esac
case "$MODE" in
  plan|train|inspect|prepare|oracle|all) need_file "$PROCESSED/index.jsonl" ;;
  evaluate) [[ "$EVAL_MODE" != full ]] || need_file "$PROCESSED/index.jsonl" ;;
esac

echo "[CONFIG] mode=$MODE run=$RUN_TAG stage1_run=$STAGE1_RUN"
echo "[CONFIG] stage1_ckpt=$STAGE1_CKPT"
echo "[CONFIG] Stage2=$S2, max_iters=$MAX_ITERS, wandb=$WANDB_MODE"
echo "[CONFIG] eval.mode=$EVAL_MODE EVAL=$EVAL PRED=$PRED"

audit_plan(){
  local plan="$S2/resolved_stage2_config.json"
  if [[ -f "$plan" ]]; then
    local now="$(mktemp)"
    "${CONFIG_CMD[@]}" --format json "${OVERRIDES[@]}" > "$now"
    if ! python - "$plan" "$now" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]));n=json.load(open(sys.argv[2]))
keys=('job.run_tag','paths.stage1_run','paths.stage1_checkpoint','paths.processed',
      'data.epoch_clips','data.seed','train.gpus','train.max_iters',
      'train.save_every','optimizer.lr','wandb.enabled','wandb.mode')
bad=[k for k in keys if p.get(k)!=n.get(k)]
for k in bad:print(f'[PLAN MISMATCH] {k}: planned={p.get(k)!r} now={n.get(k)!r}')
sys.exit(bool(bad))
PY
    then rm -f "$now"; fail 'Plan and train settings differ; use same TOML or new job.run_tag'; fi
    rm -f "$now"
  fi
  if [[ -f "$S2/stage1_checkpoint.sha256" ]]; then
    ( cd "$BASE" && sha256sum -c "$S2/stage1_checkpoint.sha256" ) || fail 'Stage1 checkpoint metadata changed'
  fi
}

inspect(){
  python - "$PROCESSED/index.jsonl" <<'PY'
import json,sys
from collections import Counter
rows=[json.loads(line) for line in open(sys.argv[1]) if line.strip()]
print('[AV2 DATASET COUNTS]',dict(Counter((r['dataset'],r['split']) for r in rows if r['dataset']=='av2')))
for sp in ('train','val'):
    seq={r['sequence'] for r in rows if r['dataset']=='av2' and r['split']==sp}
    print(f'AV2 {sp}: {len(seq)} sequences')
PY
}

plan(){
  [[ ! -e "$S2" ]] || fail "Existing Stage2 run directory: $S2 (use a NEW job.run_tag)"
  python -m cosmos3av.cli plan --stage posttrain --datasets av2 \
    --stage1-run "$STAGE1_RUN" \
    --processed-root "$PROCESSED" --init-checkpoint "$STAGE1_CKPT" --wan-vae "$WAN_VAE" \
    --outdir "$S2" --epoch-clips "$EPOCH_CLIPS" \
    --max-iters "$MAX_ITERS" --save-every "$SAVE_EVERY" --lr "$LR" \
    --gpus "$GPUS" --seed "$SEED" --wandb-mode "$WANDB_MODE" --full-preflight
  [[ -z "$CONFIG_PATH" ]] || cp "$CONFIG_PATH" "$S2/source_stage2_config.toml"
  "${CONFIG_CMD[@]}" --format json "${OVERRIDES[@]}" > "$S2/resolved_stage2_config.json"
  (cd "$BASE" && sha256sum "${STAGE1_CKPT#"$BASE/"}/model/.metadata" > "$S2/stage1_checkpoint.sha256")
  python - "$S2/run.json" "$STAGE1_CKPT" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text()); expected=Path(sys.argv[2]).resolve()
assert d['stage']=='posttrain' and d['datasets']==['av2']
assert d['action_frame']=='camera' and d['action_normalizer'] is None
assert Path(d['init_checkpoint']).resolve()==expected, 'Stage2 did not receive Stage1 DCP!'
print('[PLAN OK] Stage2 initial checkpoint is Stage1:', d['init_checkpoint'])
PY
  python -m cosmos3av.cli smoke --manifest "$S2/train.jsonl"
}

train(){
  need_file "$S2/run.json"
  audit_plan
  [[ ! -e "$S2/train.log" ]] || fail "Existing train.log: $S2/train.log (not implicitly resuming)"
  [[ ! -e "$S2_FINAL_DCP" ]] || fail "Existing final DCP: $S2_FINAL_DCP"
  echo '[TRAIN] AV2 posttraining from Stage1 DCP. Starting...' 
  unset IMAGINAIRE_OUTPUT_ROOT || true
  python -m cosmos3av.cli launch --run-dir "$S2" --framework-root "$FRAMEWORK" \
    --execute 2>&1 | tee "$S2/train.log"
  need_dcp "$S2_FINAL_DCP"
  echo "[TRAIN OK] $S2_FINAL_DCP"
}

export_model(){
  need_dcp "$S2_DCP"; need_file "$S2_CFG"
  [[ ! -e "$MODEL" ]] || fail "Existing exported model: $MODEL"
  export COSMOS_AV_MANIFEST="$S2/train.jsonl"
  export COSMOS_AV_ACTION_NORMALIZER=""
  export BASE_CHECKPOINT_PATH="$STAGE1_CKPT"
  export WAN_VAE_PATH="$WAN_VAE"
  python -m cosmos3eval.cli export --framework-root "$FRAMEWORK" \
    --dcp "$S2_DCP" --config "$S2_CFG" --out "$MODEL" --execute \
    2>&1 | tee "$S2/export_iter_${STEP}.log"
  [[ -d "$MODEL" ]] || fail "Missing exported model: $MODEL"
  printf '%s\n' "$S2_DCP" > "${MODEL}.source_dcp.txt"
  sha256sum "$S2_DCP/model/.metadata" > "${MODEL}.source_model_metadata.sha256"
  echo "[EXPORT OK] $MODEL"
}

prepare_eval(){
  if [[ -f "$EVAL/selection.jsonl" && -f "$EVAL/inference_requests.jsonl" ]]; then
    echo "[PREPARE] Using existing AV2 selection: $EVAL"
    return
  fi
  [[ ! -e "$EVAL" ]] || fail "Incomplete evaluation directory: $EVAL"
  python - "$PROCESSED" "$EVAL" "$EVAL_CLIPS" "$SEED" <<'PY'
import json,random,sys
from collections import defaultdict
from pathlib import Path
from cosmos3eval.data import read_jsonl,write_jsonl,verify_row,make_request
root,out=map(Path,sys.argv[1:3]); count=int(sys.argv[3]);seed=int(sys.argv[4])
rows=read_jsonl(root/'index.jsonl')
train={r['sequence'] for r in rows if r['dataset']=='av2' and r['split']=='train'}
val=[r for r in rows if r['dataset']=='av2' and r['split']=='val']
if not val:raise SystemExit('[ERROR] No AV2/val clips found')
overlap=train & {r['sequence'] for r in val}
if overlap:raise SystemExit(f'[ERROR] AV2 train/val sequence overlap: {sorted(overlap)[:5]}')
groups=defaultdict(list)
for r in val:groups[r['sequence']].append(r)
rng=random.Random(seed);keys=sorted(groups);rng.shuffle(keys)
for key in keys:rng.shuffle(groups[key])
chosen=[]
while len(chosen)<min(count,len(val)):
    any_added=False
    for key in keys:
        if groups[key]:
            chosen.append(groups[key].pop());any_added=True
            if len(chosen)>=min(count,len(val)):break
    if not any_added:break
for r in chosen:verify_row(r)
out.mkdir(parents=True,exist_ok=False)
write_jsonl(out/'selection.jsonl',chosen)
write_jsonl(out/'inference_requests.jsonl',[make_request(r,seed) for r in chosen])
print(f'[PREPARE OK] AV2 val clips={len(chosen)} sequences={len({r["sequence"] for r in chosen})}')
PY
}

oracle(){
  need_file "$EVAL/selection.jsonl"
  if [[ -f "$EVAL/oracle_metrics/summary.json" ]]; then
    echo "[ORACLE] Already checked: $EVAL/oracle_metrics"
    return
  fi
  python -m cosmos3eval.cli score --eval-dir "$EVAL" \
    --pred-dir "$EVAL/unused_oracle_predictions" --out "$EVAL/oracle_metrics" \
    --oracle --model gt_oracle --no-plots
}

infer(){
  need_file "$EVAL/inference_requests.jsonl"
  [[ -d "$MODEL" ]] || fail "Exported model missing: $MODEL"
  [[ ! -e "$PRED" ]] || fail "Predictions already exist: $PRED; use another job.eval_tag for a new evaluation"
  python -m cosmos3eval.cli infer --eval-dir "$EVAL" --pred-dir "$PRED" \
    --framework-root "$FRAMEWORK" --checkpoint "$MODEL" \
    --gpus "$INFER_GPUS" --seed "$SEED" --execute
  [[ -n "$(find "$PRED" -mindepth 2 -maxdepth 2 -name 'sample_outputs.json' -print -quit 2>/dev/null)" ]] || fail "No predicted samples in $PRED"
  printf '%s\n' "$MODEL" > "${PRED}.source_export.txt"
}

# Validation is read-only. Do not silently score incomplete evaluations.
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
    for cid, why in missing[:10]:
        print(f'[PRED MISSING] {cid}: {why}', file=sys.stderr)
    raise SystemExit('[ERROR] Missing or unreadable predictions. Set eval.pred_dir to a completed prediction directory or run infer for a new eval tag.')
PY_PRED_CHECK
}

score(){
  verify_predictions
  python -m cosmos3eval.cli score --eval-dir "$EVAL" --pred-dir "$PRED" \
    --model "${RUN_TAG}_iter_${STEP}" \
    --wandb-mode "$WANDB_MODE" --wandb-project "$WANDB_PROJECT" \
    --wandb-group posttrain --wandb-name "eval_${RUN_TAG}_iter_${STEP}_${EVAL_TAG}"
  python - "$PRED/metrics/summary.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); o=x['overall']
print('[STAGE2 AV2 EVALUATION]', 'clips=', o['n_clips'], 'failed=',o['n_failed'])
for k in ('rre_deg_mean','rte_m_mean','ate_m_rmse','final_position_error_m_mean'):
    print(k, o.get(k))
PY
}

render(){
  # Only render an evaluation when ALL selected clips have predictions.
  # A partial run should never silently be treated as complete.
  verify_predictions
  local selected ds cid
  selected="$(python "$SCRIPT_DIR/scripts/choose_render_clips.py" \
    --selection "$EVAL/selection.jsonl" --datasets av2 \
    --max-per-dataset "$RENDER_MAX_CLIPS" --clip-ids-json "$RENDER_CLIP_IDS_JSON")" \
    || fail "Could not select Stage2 clips for rendering"
  while IFS=$'\t' read -r ds cid; do
    [[ -n "$cid" ]] || continue
    echo "[RENDER] $ds clip=$cid (ground_height_m=$GROUND_HEIGHT_M approximate)"
    python -m cosmos3eval.cli render --eval-dir "$EVAL" --pred-dir "$PRED" \
      --clip-id "$cid" --layout "$RENDER_LAYOUT" \
      --ground-height-m "$GROUND_HEIGHT_M" --horizon-steps "$HORIZON_STEPS" \
      --fps "$RENDER_FPS" --out-dir "$PRED/qualitative/$ds"
  done <<< "$selected"
  echo "[RENDER OK] $PRED/qualitative/"
}
maybe_render(){
  if [[ "$RENDER_ENABLED" == true ]]; then render
  else echo '[RENDER SKIPPED] eval.render_enabled=false (run the render command to render separately)'; fi
}

# TOML controls the action for the *evaluate* command.
# Existing-prediction modes never invoke inference or change checkpoints.
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
  *) fail "Unexpected command: $MODE";;
esac
