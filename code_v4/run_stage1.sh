#!/usr/bin/env bash
# Run Stage1 (KITTI + Waymo) with one TOML file.
# No command-line arguments: edit only the two variables below.
set -Eeuo pipefail

# ===== USER SETTINGS =====
TOML_FILE="configs/stage1_full.toml"
RUN_STEP="all"      # all | inspect | plan | train | export | evaluate
# =========================

if (( $# != 0 )); then
  echo "[ERROR] This launcher takes no arguments. Edit TOML_FILE and RUN_STEP inside run_stage1.sh." >&2
  exit 2
fi

# Use the project directory containing this script, independent of caller's cwd.
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

[[ -f "$TOML_FILE" ]] || { echo "[ERROR] TOML not found: $TOML_FILE" >&2; exit 1; }
[[ -f "stage1_full_train_eval.sh" ]] || { echo "[ERROR] Missing stage1_full_train_eval.sh" >&2; exit 1; }
case "$RUN_STEP" in
  all|inspect|plan|train|export|evaluate) ;;
  *) echo "[ERROR] Invalid RUN_STEP: $RUN_STEP" >&2; exit 2 ;;
esac

printf '[STAGE1] TOML=%s | step=%s\n' "$TOML_FILE" "$RUN_STEP"
# `config` shows resolved parameters before any further action.
bash stage1_full_train_eval.sh --toml "$TOML_FILE" config
bash stage1_full_train_eval.sh --toml "$TOML_FILE" "$RUN_STEP"
