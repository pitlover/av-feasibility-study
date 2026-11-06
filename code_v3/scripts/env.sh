#!/usr/bin/env bash
# Source from any directory: source /path/to/cosmos3_ji/scripts/env.sh
# No implicit training, download, overwrite or directory creation.
export BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export FRAMEWORK="${FRAMEWORK:-$BASE/cosmos-framework}"
export PROCESSED="${PROCESSED:-$BASE/processed}"
export PYTHONPATH="$BASE:$FRAMEWORK${PYTHONPATH:+:$PYTHONPATH}"
echo "[ENV] BASE=$BASE"
echo "[ENV] FRAMEWORK=$FRAMEWORK"
echo "[ENV] PROCESSED=$PROCESSED"
