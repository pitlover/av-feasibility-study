#!/usr/bin/env bash
# Run Stage2 (Argoverse2), initialized from the Stage1 DCP in its TOML.
# No command-line arguments: edit only the two variables below.
set -Eeuo pipefail

# ===== USER SETTINGS =====
TOML_FILE="configs/stage2_full.toml"
RUN_STEP="all"      # all | inspect | plan | train | export | evaluate
# =========================

if (( $# != 0 )); then
  echo "[ERROR] This launcher takes no arguments. Edit TOML_FILE and RUN_STEP inside run_stage2.sh." >&2
  exit 2
fi

cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

[[ -f "$TOML_FILE" ]] || { echo "[ERROR] TOML not found: $TOML_FILE" >&2; exit 1; }
[[ -f "stage2_full_train_eval.sh" ]] || { echo "[ERROR] Missing stage2_full_train_eval.sh" >&2; exit 1; }
case "$RUN_STEP" in
  all|inspect|plan|train|export|evaluate) ;;
  *) echo "[ERROR] Invalid RUN_STEP: $RUN_STEP" >&2; exit 2 ;;
esac

printf '[STAGE2] TOML=%s | step=%s\n' "$TOML_FILE" "$RUN_STEP"
# Checks that Stage2's stage1_run and stage1_checkpoint are correct.
bash stage2_full_train_eval.sh --toml "$TOML_FILE" config
bash stage2_full_train_eval.sh --toml "$TOML_FILE" "$RUN_STEP"
