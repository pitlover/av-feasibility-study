#!/usr/bin/env bash
# Run all three variants on KITTI, Waymo, AV2 in separate output roots.
# Only local files are written. No git, no GitHub push.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-/mnt/ufs/users/jaeha-lee/share/jiiyoung/cosmos3_ji/processed_camera_fps_ablation}"

KITTI_ROOT="${KITTI_ROOT:-/mnt/ufs/dataset/open-data/KITTI/Kitti-Odometry/dataset}"
KITTI_POSES="${KITTI_POSES:-/mnt/ufs/users/jaeha-lee/share/dataset/kitti/data_odometry_poses/dataset/poses}"
KITTI_CALIB="${KITTI_CALIB:-/mnt/ufs/users/jaeha-lee/share/dataset/kitti/dataset}"
WAYMO_TRAIN_ROOT="${WAYMO_TRAIN_ROOT:-/mnt/ufs/dataset/open-data/waymo/perception/training}"
WAYMO_VAL_ROOT="${WAYMO_VAL_ROOT:-/mnt/ufs/dataset/open-data/waymo/perception/validation}"
AV2_ROOT="${AV2_ROOT:-/mnt/ufs/dataset/open-data/Argoverse2}"

BASE_PYTHON="${BASE_PYTHON:-python}"
WAYMO_PYTHON="${WAYMO_PYTHON:-$BASE_PYTHON}"
VARIANTS="${VARIANTS:-original_10hz original_native virtual_native}"
DATASETS="${DATASETS:-kitti waymo av2}"
MAX_CLIPS="${MAX_CLIPS:-1}"              # 1 for smoke, 0 for all; per dataset command
MAX_LOGS="${MAX_LOGS:-1}"                # 1 for smoke, 0 for all
STRIDE="${STRIDE:-60}"
ENCODER="${ENCODER:-local}"             # official requires compatible framework in each Python env
OVERWRITE="${OVERWRITE:-0}"
STRICT="${STRICT:-1}"

if [[ ! -f "${REPO_ROOT}/cosmos3prep/three_variants.py" ]]; then
  echo "Missing cosmos3prep/three_variants.py -- extract the ZIP into your repo root." >&2
  exit 2
fi

printf '[INPUT] REPO_ROOT=%s\n[OUTPUT] OUT_ROOT=%s\n' "$REPO_ROOT" "$OUT_ROOT"
read -r -a variant_list <<< "$VARIANTS"
read -r -a dataset_list <<< "$DATASETS"

run_prep() {
  local py="$1"; shift
  local dataset="$1"; shift
  local variant="$1"; shift
  local root="$1"; shift
  local -a opts=(
    --dataset "$dataset" --variant "$variant" --root "$root"
    --out-root "$OUT_ROOT" --stride "$STRIDE" --encoder "$ENCODER"
    --max-clips "$MAX_CLIPS" --max-logs "$MAX_LOGS"
  )
  if [[ "$OVERWRITE" == "1" ]]; then opts+=(--overwrite); fi
  if [[ "$STRICT" == "1" ]]; then opts+=(--strict); fi
  echo "[RUN] $dataset / $variant / $root"
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" "$py" -m cosmos3prep.three_variants prepare \
    "${opts[@]}" "$@"
}

for variant in "${variant_list[@]}"; do
  for dataset in "${dataset_list[@]}"; do
    case "$dataset" in
      kitti)
        run_prep "$BASE_PYTHON" kitti "$variant" "$KITTI_ROOT" \
          --poses-dir "$KITTI_POSES" --calib-root "$KITTI_CALIB" \
          --sequences 00 01 02 03 04 05 06 07 08 09 10
        ;;
      waymo)
        run_prep "$WAYMO_PYTHON" waymo "$variant" "$WAYMO_TRAIN_ROOT"
        if [[ -d "$WAYMO_VAL_ROOT" ]]; then
          run_prep "$WAYMO_PYTHON" waymo "$variant" "$WAYMO_VAL_ROOT"
        else
          echo "[warn] No Waymo val root: $WAYMO_VAL_ROOT"
        fi
        ;;
      av2)
        run_prep "$BASE_PYTHON" av2 "$variant" "$AV2_ROOT"
        ;;
      *) echo "Unknown DATASETS entry: $dataset" >&2; exit 2 ;;
    esac
  done
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" "$BASE_PYTHON" -m cosmos3prep.three_variants validate \
    --out-root "$OUT_ROOT" --variant "$variant" --max-clips 5
 done

echo "[DONE] All requested variants. Outputs: $OUT_ROOT/{original_10hz,original_native,virtual_native}"
