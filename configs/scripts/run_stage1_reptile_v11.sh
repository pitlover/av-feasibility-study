#!/usr/bin/env bash
# Stage1 no-argument entrypoint: ALL experiment configuration, including the step to run ([job] run_step =
# config|inspect|plan|train|export|prepare|oracle|infer|score|render|evaluate|all), lives in the TOML below.
# Edit only TOML_FILE; run without command-line arguments. Everything else is delegated to
# stage1_full_train_eval.sh (same checks, same output layout, Reptile when the TOML has [reptile] enabled = true).
set -Eeuo pipefail

TOML_FILE="configs/stage1_reptile_v11.toml"

(( $# == 0 )) || { echo '[ERROR] This script takes no arguments; configure the TOML file ([job] run_step selects the step).' >&2; exit 2; }
# Project root = the directory holding stage1_full_train_eval.sh: this script's own directory or an ancestor
# (the file lives in configs/scripts/ but may also be copied next to stage1_full_train_eval.sh).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$ROOT/stage1_full_train_eval.sh" && "$ROOT" != / ]]; do ROOT="$(dirname "$ROOT")"; done
[[ -f "$ROOT/stage1_full_train_eval.sh" ]] || { echo "[ERROR] stage1_full_train_eval.sh not found at or above $(dirname "${BASH_SOURCE[0]}")" >&2; exit 2; }
cd "$ROOT"
[[ -f "$TOML_FILE" ]] || { echo "[ERROR] Missing TOML: $ROOT/$TOML_FILE" >&2; exit 2; }
printf '[STAGE1] TOML=%s (step = [job] run_step)\n' "$ROOT/$TOML_FILE"
exec bash stage1_full_train_eval.sh --toml "$TOML_FILE"
