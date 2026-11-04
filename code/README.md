# KITTI + Waymo + Argoverse 2 Sensor — Ego Motion Baseline v4

기존 `ego_motion_dualmode_calib_fix` 프로젝트를 유지하며 **Argoverse 2 Sensor**를 세 번째 데이터셋으로 추가한 버전.

## 0. 가장 중요한 설정

- Input: 2 adjacent *monocular front* RGB frames + common virtual-camera intrinsic `K*`.
- Virtual camera: `640 × 384`, horizontal FOV `45°`, zero distortion, same center/orientation as each physical source camera.
- Output: `[translation_xyz (m), axis_angle_xyz (rad)]` = camera-relative SE(3) between adjacent sampled images.
- GT in manifest `frames[*].pose`: **camera-to-world** (`T_WC`), NOT vehicle trajectory and NOT frame-relative action.
- GT training label: `inv(T_WC[t]) @ T_WC[t+1]`. There is **no training-time vehicle-frame loss** in this baseline.
- 6-second trajectory: 61 images/60 relative poses, accumulate with SE(3) matrix products.
- AV2: **20 Hz original → 10 Hz** (every second image), and vehicle pose interpolated at selected camera timestamps.
- Prepared/Online: same geometric mapping function `camera.virtualize_bgr` and the same GT; only when JPEGs are stored differs.

## 1. AV2 Sensor dataset required (NOT AV2 Motion Forecasting)

Download AV2 **Sensor** `train` and `val` logs. Official: https://www.argoverse.org/av2.html and https://argoverse.github.io/user-guide/getting_started.html . Dataset access/license applies.

The project expects an extracted folder, e.g.:

```
/mnt/ufs/dataset/open-data/argoverse2/sensor/
  train/
    <log_uuid>/
      city_SE3_egovehicle.feather
      calibration/intrinsics.feather
      calibration/egovehicle_SE3_sensor.feather
      sensors/cameras/stereo_front_left/<timestamp_ns>.jpg
      ...
  val/
    <log_uuid>/ ...
```

The actual path is just a suggested location; **use the path where you extracted AV2**. Test benchmark split is skipped; it is not required for supervised GT evaluation.

Official AV2 describes the distributed camera JPEGs as **already undistorted**; although `intrinsics.feather` has historical lens coefficients `k1,k2,k3`, the stored RGB images should not be undistorted a second time. The manifest uses `distortion: [0,0,0,0,0]` for **the input JPEG** and preserves the original lens coefficients separately under `raw_lens_radial_coefficients` for provenance. This assumption should be checked against a real sample's optical projection if your download is re-encoded or transformed.

Use `stereo_front_left` to mirror OpenVO's preprocess script. Alternative: `--av2-camera ring_front_center` (portrait original). **Never mix cameras within one comparison without recording the camera choice.**

## 2. Environment

Existing `scene` Python/PyTorch environment is fine. For AV2 only this addition does *not* require compiling or installing `av2-api`; it reads Feather with pandas + pyarrow, and uses NumPy/SciPy for SE(3).

```bash
conda activate scene
python -m pip install pandas pyarrow
python -c "import pandas,pyarrow,torch,cv2,scipy; print('ok')"
```

## 3. Build raw AV2 manifest (NO virtual images saved)

```bash
python build_index.py av2 \
  --root /mnt/ufs/dataset/open-data/argoverse2/sensor \
  --av2-camera stereo_front_left \
  --av2-fps 10 \
  --max-logs-per-split 5 \
  --out indexes/av2.json
```

- `--max-logs-per-split 5`: first five train and val logs, debugging. Set `0` for ALL available logs.
- Images selected at `0,2,4,...` after sorting original 20Hz nanosecond timestamps; larger gaps are split into separate sequences. No fake frame interpolation.
- For a *selected* frame timestamp, `T_WV` is interpolated from AV2 `city_SE3_egovehicle.feather`: XYZ interpolation + quaternion SLERP; gaps/extrapolation rejected.
- `T_VC` comes from `calibration/egovehicle_SE3_sensor.feather` for the chosen camera. **This camera extrinsic is AV2's optical sensor→egovehicle transformation**, so there is no Waymo-style optical-axis remapping.
- Each global camera pose: `T_WC = T_WV @ T_VC`. Both `vehicle_pose` and `T_vehicle_camera` are preserved; the baseline loss still uses `pose` (camera).
- `K_src = intrinsics.feather` pinhole matrix. The virtual mapping uses the shared `camera.py` 640×384/45° `K*`.

Verify:

```bash
python validate_extrinsics.py --manifest indexes/av2.json
```

## 4-A. Prepared Mode (save virtual JPEGs once)

```bash
python materialize.py \
  --index indexes/av2.json \
  --out-dir processed/av2 \
  --manifest manifests/av2_prepared.json
```

Use `--overwrite` when rebuilding changed data in an existing output directory (or choose a fresh directory). `materialize` maintains poses, timestamps, extrinsics and calibration from the raw index; it only adds `prepared_path` and changes `stage` from `raw` to `prepared`.

## 4-B. Online Mode (transform during training/evaluation)

Skip `materialize.py`. Use `indexes/av2.json` directly with `--mode online`. AV2 JPEGs already exist as individual files, so unlike Waymo TFRecord there is no original-JPEG cache stage.

## 5. Train

**KITTI + Waymo + AV2 balanced joint mid-training:**

```bash
python train.py --mode prepared \
  --kitti manifests/kitti_prepared.json \
  --waymo manifests/waymo_prepared.json \
  --av2 manifests/av2_prepared.json \
  --epochs 10 --steps-per-epoch 300 --batch-size 8 \
  --num-workers 2 --out checkpoints/mid_3sets.pt
```

Uniform *dataset* probability: KITTI/Waymo/AV2 each ~1/3 regardless of raw pair counts. Existing KITTI+Waymo invocation without `--av2` still works (50:50).

**KITTI+Waymo mid-training then AV2-only post-training:**

```bash
python train.py --mode prepared \
  --av2 manifests/av2_prepared.json \
  --init-checkpoint checkpoints/pose_prepared.pt \
  --epochs 5 --steps-per-epoch 200 --batch-size 8 \
  --lr 3e-5 --num-workers 2 --out checkpoints/post_av2.pt
```

`--init-checkpoint` loads **model weights only** and resets optimizer & epoch; do not confuse with `--resume` which resumes the optimizer/epoch. This is the same *camera-relative motion task* in post-training. For zero-shot evaluation, skip AV2 post-training and evaluate `checkpoints/pose_prepared.pt` directly.

**Online mid training (no virtual images):**

```bash
python train.py --mode online \
  --kitti indexes/kitti.json --waymo indexes/waymo.json --av2 indexes/av2.json \
  --epochs 10 --steps-per-epoch 300 --batch-size 8 \
  --num-workers 2 --out checkpoints/mid_3sets_online.pt
```

Waymo's raw index still needs working TFRecord decoding or `--raw-jpeg-cache`, as in previous versions. AV2 and KITTI work directly from JPEG/PNG paths.

## 6. Evaluation and GT vs Pred visualization

```bash
python evaluate.py --mode prepared \
  --manifest manifests/av2_prepared.json \
  --checkpoint checkpoints/mid_3sets.pt \
  --split val --clip-frames 61 --max-clips 20

python visualize_prediction.py --mode prepared \
  --manifest manifests/av2_prepared.json \
  --checkpoint checkpoints/mid_3sets.pt \
  --split val --sequence-index 0 --frames 61 \
  --outdir viz/av2_gt_pred --video
```

The visualization writes `bev_comparison.png`, `first_frame_comparison.png`, `trajectory_comparison.mp4` (H.264/yuv420p, ffmpeg required), `trajectory.csv`, `metrics.json`, `poses_gt.npy`, `poses_pred.npy`. Road overlay uses a **vehicle z=0 ground reference** projected into the actual camera image, not a measured road plane. For BEV only, camera centers are used.

For online, use `--mode online --manifest indexes/av2.json` instead.

## 7. Inspect manifest

```
format_version: 3
stage: raw | prepared
camera: { K_virtual, virtual_size, virtual_hfov_deg }
sequences[*]:
  source: av2
  split: train | val
  camera_name: stereo_front_left
  original_fps: 20
  target_fps: 10
  K_src: (3,3)
  distortion: [0,0,0,0,0]   # input JPEG already undistorted
  raw_lens_radial_coefficients: [k1,k2,k3]   # NOT reapplied
  T_vehicle_camera: (4,4), optical camera→egovehicle
  vehicle_frame: av2_egovehicle
  frames[*]:
    timestamp_ns: int
    timestamp_s: relative seconds since first selected JPEG
    raw_path: original JPEG path
    pose: (4,4), optical camera→city/world
    vehicle_pose: (4,4), egovehicle→city/world
    prepared_path: only when prepared
```

## 8. Diagnostics & limitations

1. **10Hz:** The output contains ~0.100-second adjacent image timestamps and skips/splits at discontinuities. 61 frames correspond to ~6 sec (subject to measured timestamps).
2. **GT:** `pose[t] = vehicle_pose[t] @ T_vehicle_camera`; check with `validate_extrinsics.py`. Predictions are camera-relative; convert using extrinsic only when a vehicle-centric metric is needed.
3. **Intrinsic geometry:** Source camera center and orientation are unchanged by virtual mapping. All three datasets share the same virtual K*; not the same physical extrinsic.
4. **Real image test needed:** Run `materialize.py` on 1–5 real AV2 logs and inspect valid pixel coverage/road projection. Tests use synthetic AV2 tables and images; full real AV2 binaries were not available in this environment.
5. **OpenVO distinction:** The linked `preprocess_av2.py` selects `stereo_front_left` and reads AV2 sensor calibration. Its GT-export/copy block in the linked revision is commented out. This project implements pose interpolation and the canonical 10Hz output separately; it does not reproduce OpenVO's exact training/evaluation metrics.
6. **FPS ablation:** `--av2-fps 20` is supported for indexing, but the current ResNet18 has no explicit timestep encoder; do not mix 20Hz pairs with 10Hz pairs or compare without accounting for differing displacement per pair.
7. **Data access:** Only AV2 **Sensor** has RGB + 6-DoF pose; AV2 Motion Forecasting is a different dataset and is not supported by this image-based loader.
8. **K*: fixed input:** Model's intrinsic branch still receives a constant K*; this is by design and is not an intrinsic-generalization experiment.

## 9. Unit tests

```bash
PYTHONPATH=. python -m pytest -q tests/
```

## 10. Existing KITTI+Waymo commands (unchanged)

```bash
python build_index.py kitti \
  --root /mnt/ufs/dataset/open-data/KITTI/Kitti-Odometry/dataset \
  --poses-root /mnt/ufs/users/jaeha-lee/share/dataset/kitti/dataset/poses \
  --kitti-calib-root /mnt/ufs/users/jaeha-lee/share/dataset/kitti/dataset/sequences \
  --kitti-frame velodyne --val-seqs 09,10 --out indexes/kitti.json

# Run in waymo-prep env if needed:
python build_index.py waymo \
  --root /mnt/ufs/dataset/open-data/waymo/perception \
  --max-files-per-split 5 --raw-jpeg-cache cache/waymo_original_jpeg \
  --out indexes/waymo.json

python materialize.py --index indexes/kitti.json \
  --out-dir processed/kitti --manifest manifests/kitti_prepared.json
python materialize.py --index indexes/waymo.json \
  --out-dir processed/waymo --manifest manifests/waymo_prepared.json
```

Source links:
- https://argoverse.github.io/user-guide/datasets/sensor.html
- https://arxiv.org/html/2602.19035v2
- https://github.com/PhucNDA/OpenVO/blob/main/data/preprocess_av2.py
