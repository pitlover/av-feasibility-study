#!/usr/bin/env bash
# nuScenes CAM_FRONT, COMMON virtual intrinsic (640x384 HFOV45), native nominal 12Hz.
# SMOKE=1 bash scripts/run_nuscenes_virtual_native.sh
# SMOKE=0 bash scripts/run_nuscenes_virtual_native.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
NUSCENES_ROOT="${NUSCENES_ROOT:-/mnt/ufs/users/jaeha-lee/data/finetuned_data/nuscenes_source}"
OUT_ROOT="${OUT_ROOT:-/mnt/ufs/users/jaeha-lee/share/jiiyoung/cosmos3_ji/processed_virtual_rawhz}"
EXTRA=()
if [[ "${SMOKE:-0}" == "1" ]]; then EXTRA=(--max-logs 1 --max-clips 1); fi
python -m cosmos3prep.three_variants prepare \
  --dataset nuscenes \
  --variant virtual_native \
  --root "$NUSCENES_ROOT" \
  --camera CAM_FRONT \
  --out-root "$OUT_ROOT" \
  --virtual-width 640 \
  --virtual-height 384 \
  --virtual-hfov-deg 45 \
  --encoder local \
  --strict \
  "${EXTRA[@]}"
python -m cosmos3prep.three_variants validate \
  --out-root "$OUT_ROOT" \
  --variant virtual_native \
  --only-dataset nuscenes \
  --max-clips "${VALIDATE_MAX_CLIPS:-3}"
