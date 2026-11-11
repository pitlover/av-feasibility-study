# Cosmos3-Edge AV Inverse Dynamics — Complete source (V2-compatible)

This ZIP contains **actual Python source modules**, not a deployment tool or a patch script.

The source combines the `fix/sync-v2-cli-stage-training` implementation with the
subsequently verified V2 metadata compatibility correction. The Cosmos3
training/evaluation modules were reconstructed from previously retrieved source
where the original server workspace was inaccessible. **This is not a byte-for-byte
copy of the user's live, tested server folder** and must be regression-tested
before replacing it. The user's own server reported 6 V2 compatibility tests
passing and three AV2 GT/action roundtrips passing; this archive was separately
validated with its bundled tests.

## Direct install (copy source files, preserve data/checkpoints)

Unzip **outside** the current working directory. Back up the current Python
modules and scripts before replacing them. Copy only `cosmos3prep`,
`cosmos3virtual`, `cosmos3av`, `cosmos3eval`, `scripts`, `tests`, and the root
Python/requirements files. Do **not** replace any `processed*`, `runs*`,
`checkpoints*`, `exports`, `evaluation`, `normalizers`, or `cosmos-framework`.

### Environment

```bash
conda activate cosmos
cd /mnt/ufs/users/jaeha-lee/share/jiiyoung/cosmos3_ji
export BASE="$PWD"
export FRAMEWORK="$BASE/cosmos-framework"
export VIRTUAL="$BASE/processed_virtual_native_v2"
export PYTHONPATH="$BASE:$FRAMEWORK"
```

### Validate existing data

```bash
python -m pytest -q tests/test_v2_metadata_compat.py
python -m pytest -q tests/test_v2_cli_sync.py
python -m cosmos3virtual.cli validate --out "$VIRTUAL" --max-clips 3
python -m cosmos3av.cli timing --processed-root "$VIRTUAL" --dataset all --max-clips 10
```

### Preprocessing

```bash
python -m cosmos3virtual.cli kitti \
    --root /mnt/ufs/dataset/open-data/KITTI/Kitti-Odometry/dataset \
    --poses-dir /mnt/ufs/users/jaeha-lee/share/dataset/kitti/data_odometry_poses/dataset/poses \
    --calib-root /mnt/ufs/users/jaeha-lee/share/dataset/kitti/dataset \
    --sequences 00 01 02 03 04 05 06 07 08 09 10 --camera image_2 \
    --time-mode native --stride 60 --out "$VIRTUAL" --width 640 --height 384 --hfov 45 --strict

/home/ad-group/miniconda3/envs/waymo-prep/bin/python -m cosmos3virtual.cli waymo \
    --root /mnt/ufs/dataset/open-data/waymo/perception/training \
    --camera FRONT --time-mode native --stride 60 --out "$VIRTUAL" \
    --width 640 --height 384 --hfov 45 --strict

python -m cosmos3virtual.cli av2 \
    --root /mnt/ufs/dataset/open-data/Argoverse2/train \
    --camera stereo_front_left --time-mode native --stride 60 --out "$VIRTUAL" \
    --width 640 --height 384 --hfov 45 --strict
```

Use the actual AV2 train/val directory layout. Existing completed clips are
skipped by the preprocessor when overwrite is not requested.

### Stage1 and Stage2 (EXPERIMENTAL Cosmos3-Edge SFT)

```bash
export EDGE_DCP=/path/to/Cosmos3-Edge
export VAE=/path/to/Wan2.2_VAE.pth
python -m cosmos3av.cli plan --stage pretrain --datasets kitti waymo \
    --weights kitti:1,waymo:1 --processed-root "$VIRTUAL" \
    --init-checkpoint "$EDGE_DCP" --wan-vae "$VAE" \
    --outdir "$BASE/runs_virtual/stage1_joint_300_NEW" \
    --epoch-clips 500 --max-iters 300 --save-every 50 --gpus 4 --full-preflight
python -m cosmos3av.cli launch --run-dir "$BASE/runs_virtual/stage1_joint_300_NEW" \
    --framework-root "$FRAMEWORK" --execute
```

Stage2 must use an actual Stage1 DCP from the new Stage1 run and the
`--stage1-run` flag to check matching camera GT contracts.

The original user-reported Stage1/Stage2 checkpoints had action normalizer
`None`: **do not run denormalize-pred on them**. A new optional custom
translation-only normalizer can be fitted with `cosmos3virtual.cli
fit-normalizer`, but it is **not** proven identical to past server code.

### Evaluation

```bash
python -m cosmos3eval.cli prepare --processed-root "$VIRTUAL" --datasets av2 \
  --split val --max-clips-per-dataset 2 --out "$BASE/evaluation/av2_val_v2_check"
python -m cosmos3eval.cli score \
  --eval-dir "$BASE/evaluation/av2_val_v2_check" \
  --pred-dir "$BASE/evaluation/pred_stage2_av2" --model stage2_av2_raw
python -m cosmos3eval.cli render \
  --eval-dir "$BASE/evaluation/av2_val_v2_check" \
  --pred-dir "$BASE/evaluation/pred_stage2_av2" \
  --layout both --horizon-steps 30 --ego-height-m 1.7 \
  --out-dir "$BASE/evaluation/camera_viz_v2_check"
```

- Action data: `61` RGB frames / `60 x 9` camera SE(3) actions at 10Hz.
- Virtual camera: `640x384`, HFOV 45°, optical X-right / Y-down / Z-forward.
- V2 `pose_local_file` and `format=cosmos3_virtual_camera_v2` are resolved
  explicitly. Missing V2 `action_encoding` is accepted only for known V2
  metadata; unknown explicit encodings still fail.
- RRE/RTE are per-step relative SE(3) errors; ATE is first-frame-local RMSE.
- Camera image road trajectory is an **approximate flat-road projection**.
- Official framework checkout and model weights are NOT included in the ZIP.
