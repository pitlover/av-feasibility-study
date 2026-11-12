#!/usr/bin/env bash
# nuScenes CAM_FRONT, ORIGINAL intrinsic/pixels, native nominal 12Hz.
# SMOKE=1 bash scripts/run_nuscenes_original_native.sh  # first-scene one-clip smoke test
# SMOKE=0 bash scripts/run_nuscenes_original_native.sh  # process all 850 scenes
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
NUSCENES_ROOT="${NUSCENES_ROOT:-/mnt/ufs/users/jaeha-lee/data/finetuned_data/nuscenes_source}"
OUT_ROOT="${OUT_ROOT:-/mnt/ufs/users/jaeha-lee/share/jiiyoung/cosmos3_ji/processed_raw_rawhz}"
EXTRA=()
if [[ "${SMOKE:-0}" == "1" ]]; then EXTRA=(--max-logs 1 --max-clips 1); fi
python -m cosmos3prep.three_variants prepare \
  --dataset nuscenes \
  --variant original_native \
  --root "$NUSCENES_ROOT" \
  --camera CAM_FRONT \
  --out-root "$OUT_ROOT" \
  --encoder local \
  --strict \
  "${EXTRA[@]}"
python -m cosmos3prep.three_variants validate \
  --out-root "$OUT_ROOT" \
  --variant original_native \
  --only-dataset nuscenes \
  --max-clips "${VALIDATE_MAX_CLIPS:-3}"
