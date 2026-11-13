#!/usr/bin/env bash
# Cosmos3 Stage-1 launcher (v5): launch from configs/scripts/.
# This wrapper runs the existing stage1_full_train_eval.sh from the PROJECT ROOT,
# so [paths].base = "." and relative paths in the TOML work correctly.
# No GitHub writes; preprocessing sources are read-only.
set -Eeuo pipefail

# ============== EDIT HERE ==============
# TOML is relative to the repository root (not configs/scripts/).
# KITTI + Waymo + AV2; virtual camera, 10 Hz:
TOML_FILE="configs/stage1_1_virtual_10hz_reptile_v3.toml

# Default: full pipeline (inspect -> plan -> train -> export -> evaluate).
# Override with an argument: config, inspect, plan, train, export, evaluate.
DEFAULT_STEP="all"
# =======================================

# The script is stored at <repository>/configs/scripts/run_stage1_multidataset.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MAIN_SCRIPT="$ROOT/stage1_full_train_eval.sh"

if [[ ! -f "$MAIN_SCRIPT" ]]; then
  echo "[ERROR] Missing Stage-1 script: $MAIN_SCRIPT" >&2
  exit 2
fi
case "$TOML_FILE" in
  /*) TOML_PATH="$TOML_FILE" ;;
  *)  TOML_PATH="$ROOT/$TOML_FILE" ;;
esac
if [[ ! -f "$TOML_PATH" ]]; then
  echo "[ERROR] Missing TOML file: $TOML_PATH" >&2
  exit 2
fi

RUN_STEP="$DEFAULT_STEP"
case "${1:-}" in
  -h|--help)
    cat <<'EOF'
Usage: bash run_stage1_multidataset.sh [config|inspect|plan|train|export|evaluate|all] [section.key=value ...]
TOML and default step are configured at the top of this script.
Examples:
  bash run_stage1_multidataset.sh config
  bash run_stage1_multidataset.sh inspect
  bash run_stage1_multidataset.sh plan
  bash run_stage1_multidataset.sh train
  bash run_stage1_multidataset.sh plan job.run_tag=s1_smoke train.max_iters=10 train.save_every=10 data.epoch_clips=120
  bash run_stage1_multidataset.sh train job.run_tag=s1_smoke train.max_iters=10 train.save_every=10 data.epoch_clips=120
EOF
    exit 0 ;;
  config|inspect|plan|train|export|prepare|oracle|infer|score|render|evaluate|all)
    RUN_STEP="$1"
    shift ;;
  ""|*=*)
    # No explicit command: use DEFAULT_STEP; allow dotted overrides.
    ;;
  *)
    echo "[ERROR] Unknown command: $1" >&2
    echo "[HINT] Run bash $(basename "$0") --help" >&2
    exit 2 ;;
esac

cd "$ROOT"
printf '[LAUNCH] project root: %s\n' "$ROOT"
printf '[LAUNCH] TOML:         %s\n' "$TOML_PATH"
printf '[LAUNCH] step:         %s\n' "$RUN_STEP"
printf '[LAUNCH] cwd:          %s\n' "$PWD"
exec bash "$MAIN_SCRIPT" --toml "$TOML_PATH" "$RUN_STEP" "$@"
