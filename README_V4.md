# Cosmos3-Edge AV ID — code_v4 (camera-relative + virtual camera)

This is an **implemented reconstruction** using the uploaded `code_v3` snapshot and the discussion about the recent AV2 camera-frame visualization. It is not a byte-for-byte restoration of unpublished server-side edits. The existing `code_v3` remains untouched. **No real AV2, Cosmos3 GPU training, export or inference has been executed here**.

## What changed

| Component | Updates |
|---|---|
| `cosmos3virtual` (new) | Virtual pinhole RGB (640 × 384, 45° horizontal FOV, `fx=fy=772.548`, `cx=320`, `cy=192`), inverse undistort/reproject map, unchanged physical extrinsic, camera pose composition and projection helpers |
| `cosmos3prep` | `--image-mode virtual`, `--action-frame camera` defaults. `world_camera = world_ego @ T_ego_camera`; generate GT actions from camera-first-frame-local transforms, persist both ego and camera poses plus explicit convention metadata. Legacy `--image-mode letterbox --action-frame ego` still supported. 10Hz / 61-frame / 60×9 action structure remains. |
| `cosmos3av` | Read training manifests and reject mixing camera and ego actions within a training run. Existing Edge SFT/Stage1/Stage2 launcher remains. No claim that NVIDIA checkpoint output normalization was validated. |
| `cosmos3eval` | Select GT pose from action frame; camera BEV X-right/Z-forward and matching numeric metrics; render camera-image future-road-path overlay with side BEV and telemetry. Legacy render is preserved. `diagnose` outputs per-step rotation-vector Y signed drift CSV. |
| `tests` | Tests for virtual K, no-op remap, extrinsic frame conversion/roundtrip, yaw drift/road projection, plus retained code_v3 tests. |

### Geometry: essential contracts

- Coordinates for v4 **camera actions**: optical `X right, Y down, Z forward`.
- Homogeneous matrices use **T_target_source**, column vectors. `T_world_camera = T_world_ego @ T_ego_camera`.
- New virtual camera changes *intrinsics only*: it **does not** assign a new camera center, orientation, or extrinsic. It applies distortion rectification and pinhole remapping only.
- GT actions use `backward_framewise` 9D = `translation(3) + column-oriented rot6d(6)`; 61 RGB frames at nominal 10Hz and 60 actions. Framewise action translation is in the **previous camera** frame; 6DoF is reconstructed by SE(3) integration.
- Older ego-action clips and new camera-action clips **cannot be interpreted as the same targets**. Model and scoring data must have matching metadata and action normalizer.
- Camera image reprojection: future camera-anchored *approximate ground points* are mapped into the **current frame** using the predicted/GT camera trajectory, then K_virtual. The camera center itself must **not** be projected as a road point. Current implementation assumes an approximately level camera and a constant road height beneath it. A camera with significant pitch / non-level road needs a calibrated ground plane or appropriate ego-ground projection. For AV2, visually inspect this condition.
- The mean step RRE/RTE and cumulative trajectory errors are not interchangeable. `diagnose` reports the signed **rotation-vector Y component**, which is a yaw proxy for near-planar motion, not necessarily the true full 3D yaw Euler angle.

## Install / initial inspection

Keep the `code_v4/` directory intact under the project root. Set `BASE` to the existing `cosmos3_ji` directory if needed:

```bash
cd /path/to/project/code_v4
export BASE=/mnt/ufs/users/jaeha-lee/share/jiiyoung/cosmos3_ji
source scripts/env.sh
python -m pytest -q --import-mode=importlib tests
```

**Do not overwrite** the original `processed`, `evaluation`, `runs`, or `exports`. `scripts/env.sh` uses `processed_v4` by default, keeping v4 outputs isolated.

## Preprocessing: AV2 Sensor

Determine the actual AV2 directory layout first; your data root is:

```bash
find /mnt/ufs/dataset/open-data/Argoverse2 -name city_SE3_egovehicle.feather -print -quit
```

Use the containing `sensor/train` or `sensor/val` parent, or their enclosing directory, depending on the actual layout. Example (adjust the subdirectory to the observed path):

```bash
python -m cosmos3prep.cli av2 \
  --root /mnt/ufs/dataset/open-data/Argoverse2/sensor/train \
  --camera stereo_front_left --out "$PROCESSED" \
  --image-mode virtual --virtual-hfov-deg 45 \
  --width 640 --height 384 --action-frame camera \
  --encoder local --convention backward_framewise \
  --max-logs 1 --max-clips 2 --strict

python -m cosmos3prep.cli validate --out "$PROCESSED" --max-clips 2
```

Run train and val into the **same isolated** `processed_v4` root; the index rebuild will include both. For Waymo FRONT and KITTI `image_2`, the same v4 defaults apply. The source adapters retain v3 camera data/pose assumptions; verify actual camera extrinsics and distortion in the source files before full-scale training.

## Eval / inference / visual rendering

Use a held-out v4 val split; do NOT evaluate a camera-action checkpoint on the ego-action data from v3.

```bash
export EVAL="$BASE/evaluation/av2_val_virtual_v4"
export PRED="$BASE/evaluation/pred_stage2_av2_v4"
python -m cosmos3eval.cli prepare --processed-root "$PROCESSED" \
  --datasets av2 --split val --max-clips-per-dataset 2 --out "$EVAL"

# Run actual model inference separately, with the exported matching checkpoint.
# Example once real predictions exist in $PRED/CLIP_ID/sample_outputs.json:
python -m cosmos3eval.cli score --eval-dir "$EVAL" --pred-dir "$PRED" \
  --out "$PRED/metrics" --model stage2_av2_v4
python -m cosmos3eval.cli render --eval-dir "$EVAL" --pred-dir "$PRED" \
  --out-dir "$PRED/camera_viz" --mode camera --max-clips 2 \
  --ground-height-m 1.6 --max-depth-m 50
python -m cosmos3eval.cli diagnose --eval-dir "$EVAL" --pred-dir "$PRED" \
  --out-dir "$PRED/diagnostics"
```

`--ground-height-m 1.6` is **illustrative**; use the calibrated camera height to the road for real overlay. Without the parameter, the implementation tries the extrinsic vehicle-Z offset and refuses implausible heights. The overlay is an *approximation* until road geometry is verified. Export/Infer and Stage1/Stage2 commands remain in the original `README.md` and module CLIs; they have NOT been GPU validated in this reconstructed v4.

### Known limitations / do not silently assume

1. A tiny GT Action/Pose roundtrip error checks internal encode/decode consistency only; it cannot prove the physical extrinsics, camera optical axes, or actual NVIDIA prediction normalization are correct.
2. AV2 video sampling selects images near a 10Hz grid; vehicle poses are interpolated to camera timestamps by the source adapter. Check AV2 source layout, recording/pose time gap and source calibration first.
3. Input remap changes source viewpoint FOV but not camera center or rotation; distortion coefficients rely on dataset-specific OpenCV representation. Validate using a calibration target or known lane geometry.
4. Real AV2 and GPU checkpoints were not available to this build. No new model weights are supplied; v4 needs fresh training (or a correctly matched camera-action checkpoint) before its predictions can be trusted.
5. Keep all dataset GT/evaluation action conventions identical across train, val and inference. Legacy `code_v3` outputs are not automatically migrated to camera-relative targets.
