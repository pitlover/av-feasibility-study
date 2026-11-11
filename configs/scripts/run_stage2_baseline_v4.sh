#!/usr/bin/env bash
# Stage2 BASELINE (no Stage1) no-argument entrypoint: Cosmos3-Edge DCP -> AV2 post-training -> evaluation.
# ALL configuration, including the step to run ([job] run_step =
# config|inspect|plan|train|export|prepare|oracle|infer|score|render|evaluate|all), lives in the TOML below.
# Edit only TOML_FILE; run without command-line arguments, e.g.
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash configs/scripts/run_stage2_baseline_v1.sh
#
# Why this script does plan/train/export itself: stage2_full_train_eval.sh requires a Stage1 run
# (stage1_run/run.json with stage=pretrain, checkpoint under stage1_run/outputs) before plan/train/export.
# Here the init checkpoint is the original Cosmos3-Edge DCP, so those steps call the same cosmos3av/cosmos3eval
# commands directly, with the only difference that `plan` gets no --stage1-run. Evaluation steps never touch Stage1,
# so inspect/prepare/oracle/infer/score/render/evaluate are delegated to stage2_full_train_eval.sh unchanged
# (same AV2 val selection, metrics and rendering as the Stage1 -> Stage2 runs).
# In the TOML, paths.stage1_checkpoint holds the Edge DCP and paths.stage1_run must be empty.
set -Eeuo pipefail

TOML_FILE="configs/stage2_baseline_v4.toml"

(( $# == 0 )) || { echo '[ERROR] This script takes no arguments; configure the TOML file ([job] run_step selects the step).' >&2; exit 2; }
# Project root = the directory holding stage2_full_train_eval.sh: this script's own directory or an ancestor.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$ROOT/stage2_full_train_eval.sh" && "$ROOT" != / ]]; do ROOT="$(dirname "$ROOT")"; done
[[ -f "$ROOT/stage2_full_train_eval.sh" ]] || { echo "[ERROR] stage2_full_train_eval.sh not found at or above $(dirname "${BASH_SOURCE[0]}")" >&2; exit 2; }
cd "$ROOT"
[[ -f "$TOML_FILE" ]] || { echo "[ERROR] Missing TOML: $ROOT/$TOML_FILE" >&2; exit 2; }
CONFIG_PATH="$ROOT/$TOML_FILE"
HELPER="$ROOT/scripts/stage2_toml_config.py"
CONFIG_CMD=(python "$HELPER" --toml "$CONFIG_PATH")

CFG_OUT="$(mktemp)"
trap 'rm -f "$CFG_OUT"' EXIT
"${CONFIG_CMD[@]}" --format env0 > "$CFG_OUT" || exit 2
while IFS= read -r -d '' key && IFS= read -r -d '' value; do
  printf -v "$key" '%s' "$value"
done < "$CFG_OUT"
rm -f "$CFG_OUT"
MODE="$RUN_STEP"
case "$MODE" in config|inspect|plan|train|export|prepare|oracle|infer|score|render|evaluate|all) ;; *) echo "[ERROR] Bad job.run_step: $MODE" >&2; exit 2;; esac
printf '[STAGE2 BASELINE] TOML=%s | STEP=%s\n' "$CONFIG_PATH" "$MODE"

fail(){ echo "[ERROR] $*" >&2; exit 1; }
need_file(){ [[ -f "$1" ]] || fail "Missing file: $1"; }
need_dcp(){ need_file "$1/model/.metadata"; need_file "$1/trainer/.metadata"; }
delegate(){ echo "[DELEGATE] stage2_full_train_eval.sh $1"; bash "$ROOT/stage2_full_train_eval.sh" --toml "$CONFIG_PATH" "$1"; }

BASE="$(cd "$BASE" && pwd)"
for var in FRAMEWORK PROCESSED STAGE1_CKPT WAN_VAE; do
  val="${!var}"
  if [[ -n "$val" && "$val" != /* ]]; then printf -v "$var" '%s' "$BASE/$val"; fi
done
INIT_DCP="$STAGE1_CKPT"   # baseline: paths.stage1_checkpoint = Cosmos3-Edge DCP
[[ -z "$STAGE1_RUN" ]] || fail "Baseline TOML must have paths.stage1_run = \"\" (got '$STAGE1_RUN'); use run_stage2_reptile_v*.sh for Stage1 runs"
[[ -n "$WAN_VAE" ]] || fail 'Baseline TOML must set paths.wan_vae (there is no Stage1 run.json to read it from)'
# stage2_full_train_eval.sh (used for evaluation) always writes under BASE/{runs_stage2,exports,evaluation}.
[[ "$RUNS_ROOT" == runs_stage2 && "$EXPORTS_ROOT" == exports && "$EVALUATION_ROOT" == evaluation ]] \
  || fail 'Keep paths.run_root/exports_root/evaluation_root at runs_stage2/exports/evaluation (evaluation is delegated)'

S2="$BASE/runs_stage2/$RUN_TAG"
S2_OUTPUT="$S2/outputs/cosmos3_av_id/posttrain/$RUN_TAG"
S2_FINAL_DCP="$S2_OUTPUT/checkpoints/$(printf 'iter_%09d' "$MAX_ITERS")"
S2_DCP="$S2_OUTPUT/checkpoints/$(printf 'iter_%09d' "$STEP")"
S2_CFG="$S2_OUTPUT/config.yaml"
MODEL="$BASE/exports/${RUN_TAG}_iter_${STEP}"
export PYTHONPATH="$BASE:$FRAMEWORK${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$MODE" == config ]]; then
  echo "[OUTPUT] stage2_run=$S2"
  echo "[OUTPUT] final_dcp=$S2_FINAL_DCP"
  echo "[OUTPUT] export=$MODEL"
  echo "[OUTPUT] pred_dir=$BASE/evaluation/pred_${RUN_TAG}_iter_${STEP}_${EVAL_TAG}"
  echo '[CONFIG] Parsed TOML values:'
  "${CONFIG_CMD[@]}" --format json
  echo "[CONFIG] Stage2 init checkpoint (Cosmos3-Edge, no Stage1): $INIT_DCP"
  exit 0
fi

case "$MODE" in
  plan|train|export|all)
    [[ -d "$INIT_DCP" ]] || fail "Cosmos3-Edge init checkpoint missing: $INIT_DCP"
    need_file "$WAN_VAE"
    need_file "$FRAMEWORK/cosmos_framework/scripts/train.py"
    ;;
esac
case "$MODE" in plan|all) need_file "$PROCESSED/index.jsonl" ;; esac

audit_plan(){
  local plan="$S2/resolved_stage2_config.json" now
  if [[ -f "$plan" ]]; then
    now="$(mktemp)"
    "${CONFIG_CMD[@]}" --format json > "$now"
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
  if [[ -f "$S2/init_checkpoint.sha256" ]]; then
    sha256sum -c "$S2/init_checkpoint.sha256" || fail 'Init (Cosmos3-Edge) checkpoint metadata changed'
  fi
}

plan(){
  [[ ! -e "$S2" ]] || fail "Existing Stage2 run directory: $S2 (use a NEW job.run_tag)"
  echo "[PLAN] AV2 post-training initialized from Cosmos3-Edge (no --stage1-run): $INIT_DCP"
  python -m cosmos3av.cli plan --stage posttrain --datasets av2 \
    --processed-root "$PROCESSED" --init-checkpoint "$INIT_DCP" --wan-vae "$WAN_VAE" \
    --outdir "$S2" --epoch-clips "$EPOCH_CLIPS" \
    --max-iters "$MAX_ITERS" --save-every "$SAVE_EVERY" --lr "$LR" \
    --gpus "$GPUS" --seed "$SEED" --wandb-mode "$WANDB_MODE" --full-preflight
  cp "$CONFIG_PATH" "$S2/source_stage2_config.toml"
  "${CONFIG_CMD[@]}" --format json > "$S2/resolved_stage2_config.json"
  if [[ -f "$INIT_DCP/model/.metadata" ]]; then
    sha256sum "$INIT_DCP/model/.metadata" > "$S2/init_checkpoint.sha256"
  fi
  python - "$S2/run.json" "$INIT_DCP" <<'PY'
import json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text()); expected=Path(sys.argv[2]).resolve()
assert d['stage']=='posttrain' and d['datasets']==['av2']
assert d['action_frame']=='camera' and d['action_normalizer'] is None
assert d.get('stage1_run') is None, 'Baseline plan unexpectedly has a Stage1 run'
assert Path(d['init_checkpoint']).resolve()==expected, 'Stage2 did not receive the Cosmos3-Edge DCP!'
print('[PLAN OK] Stage2 initial checkpoint is Cosmos3-Edge (no Stage1):', d['init_checkpoint'])
PY
  python -m cosmos3av.cli smoke --manifest "$S2/train.jsonl"
}

train(){
  need_file "$S2/run.json"
  audit_plan
  [[ ! -e "$S2/train.log" ]] || fail "Existing train.log: $S2/train.log (not implicitly resuming)"
  [[ ! -e "$S2_FINAL_DCP" ]] || fail "Existing final DCP: $S2_FINAL_DCP"
  echo '[TRAIN] AV2 posttraining from Cosmos3-Edge DCP (no Stage1). Starting...'
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
  export BASE_CHECKPOINT_PATH="$INIT_DCP"
  export WAN_VAE_PATH="$WAN_VAE"
  mkdir -p "$(dirname "$MODEL")"
  python -m cosmos3eval.cli export --framework-root "$FRAMEWORK" \
    --dcp "$S2_DCP" --config "$S2_CFG" --out "$MODEL" --execute \
    2>&1 | tee "$S2/export_iter_${STEP}.log"
  [[ -d "$MODEL" ]] || fail "Missing exported model: $MODEL"
  printf '%s\n' "$S2_DCP" > "${MODEL}.source_dcp.txt"
  sha256sum "$S2_DCP/model/.metadata" > "${MODEL}.source_model_metadata.sha256"
  echo "[EXPORT OK] $MODEL"
}

case "$MODE" in
  plan) plan;;
  train) train;;
  export) export_model;;
  inspect|prepare|oracle|infer|score|render|evaluate) delegate "$MODE";;
  all)
    [[ "$EVAL_MODE" == full ]] || fail 'all requires eval.mode="full"'
    delegate inspect; plan; train; export_model; delegate evaluate;;
esac
