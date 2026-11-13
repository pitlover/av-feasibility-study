#!/usr/bin/env bash
# Stage-2 entrypoint: choose TOML here, choose operation in [job].run_step.
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Change only this line to select a different TOML.
TOML="$PROJECT_ROOT/configs/stage2_multisource_4_original_10hz_v3_3.toml"

ORIGINAL_SCRIPT="$PROJECT_ROOT/stage2_full_train_eval.sh"
PYTHON_BIN="${PYTHON_BIN:-python3}"

[[ -f "$TOML" ]] || { echo "[ERROR] Missing TOML: $TOML" >&2; exit 1; }
[[ -f "$ORIGINAL_SCRIPT" ]] || { echo "[ERROR] Missing Stage-2 script: $ORIGINAL_SCRIPT" >&2; exit 1; }

RUN_STEP="$("$PYTHON_BIN" - "$TOML" <<'PY'
import sys
import tomllib
from pathlib import Path

config = tomllib.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
step = config.get("job", {}).get("run_step", "config")
allowed = {"config", "inspect", "plan", "train", "export", "prepare", "oracle", "infer", "score", "render", "evaluate", "all"}
if step not in allowed:
    raise SystemExit(f"[ERROR] Invalid [job].run_step: {step!r}")
print(step)
PY
)"

printf '[TOML] %s\n[RUN STEP] %s\n' "$TOML" "$RUN_STEP"
cd "$PROJECT_ROOT"
exec bash "$ORIGINAL_SCRIPT" --toml "$TOML" "$RUN_STEP"
