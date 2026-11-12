#!/usr/bin/env bash
# Stage2 no-argument entrypoint for the Stage1 Reptile run of configs/stage1_reptile_v10.toml.
# ALL configuration, including the step to run ([job] run_step =
# config|inspect|plan|train|export|prepare|oracle|infer|score|render|evaluate|all), lives in the TOML below.
# Edit only TOML_FILE; run without command-line arguments. Everything else is delegated to
# stage2_full_train_eval.sh (same Stage1 lineage checks, same output layout).
set -Eeuo pipefail

TOML_FILE="configs/stage2_reptile_v13.1.toml"

(( $# == 0 )) || { echo '[ERROR] This script takes no arguments; configure the TOML file ([job] run_step selects the step).' >&2; exit 2; }
# Project root = the directory holding stage2_full_train_eval.sh: this script's own directory or an ancestor
# (the file lives in configs/scripts/ but may also be copied next to stage2_full_train_eval.sh).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$ROOT/stage2_full_train_eval.sh" && "$ROOT" != / ]]; do ROOT="$(dirname "$ROOT")"; done
[[ -f "$ROOT/stage2_full_train_eval.sh" ]] || { echo "[ERROR] stage2_full_train_eval.sh not found at or above $(dirname "${BASH_SOURCE[0]}")" >&2; exit 2; }
cd "$ROOT"
[[ -f "$TOML_FILE" ]] || { echo "[ERROR] Missing TOML: $ROOT/$TOML_FILE" >&2; exit 2; }
# stage2_full_train_eval.sh takes the step as an argument, so read [job] run_step with the same strict parser.
RUN_STEP="$(python scripts/stage2_toml_config.py --toml "$TOML_FILE" --format json \
  | python -c 'import json,sys; print(json.load(sys.stdin)["job.run_step"])')" || exit 2
printf '[STAGE2] TOML=%s | STEP=%s\n' "$ROOT/$TOML_FILE" "$RUN_STEP"
exec bash stage2_full_train_eval.sh --toml "$TOML_FILE" "$RUN_STEP"
