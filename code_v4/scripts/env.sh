#!/usr/bin/env bash
# Source while working inside the existing Cosmos3 project.
export CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BASE="${BASE:-$CODE_ROOT}"
export FRAMEWORK="${FRAMEWORK:-$BASE/cosmos-framework}"
export VIRTUAL="${VIRTUAL:-$BASE/processed_virtual_native_v2}"
export PROCESSED="${PROCESSED:-$VIRTUAL}"
export PYTHONPATH="$CODE_ROOT:$FRAMEWORK${PYTHONPATH:+:$PYTHONPATH}"
echo "[ENV] CODE_ROOT=$CODE_ROOT BASE=$BASE VIRTUAL=$VIRTUAL FRAMEWORK=$FRAMEWORK"
