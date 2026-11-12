# nuScenes CAM_FRONT native 12Hz preprocessing: 2 variants

**Source checkout:** `pitlover/av-feasibility-study` branch `feat/prepare_nuscenes` with the previously installed `cosmos3prep/three_variants.py`.

**No GitHub writes.** This ZIP contains local source files only. `cosmos3prep/three_variants.py` is a replacement version of the previously supplied 3-variant script with nuScenes support added, and `cosmos3prep/nuscenes_adapter.py` is new. The three existing KITTI/Waymo/AV2 variants remain available.

## Input dataset

```text
/mnt/ufs/users/jaeha-lee/data/finetuned_data/nuscenes_source/
├── samples/CAM_FRONT/      # 2Hz keyframes
├── sweeps/CAM_FRONT/       # intermediate camera frames (full camera stream ~12Hz)
├── maps/
└── v1.0-trainval/
    ├── sample_data.json
    ├── ego_pose.json
    ├── calibrated_sensor.json
    ├── sample.json
    ├── scene.json
    └── sensor.json
```

- **Camera:** `CAM_FRONT` (original 1600×900 nominal).
- **FPS:** native, nominal 12 Hz. No RGB interpolation, frame duplication, or 10Hz downsampling. Missing/gapped camera records break frame runs rather than bridging discontinuities.
- **Keyframes only are insufficient:** `samples` are around 2Hz. This code follows `sample_data.next` from each scene's first CAM_FRONT keyframe to its last, including `sweeps` images.
- **Ground truth:** each CAM_FRONT sample_data has its own `ego_pose_token`, `calibrated_sensor_token`, timestamp. Motion is camera-relative from `T_global_camera = T_global_ego @ T_ego_camera`; orientation quaternions are `(w,x,y,z)` in source JSON.
- **Calibration:** nuScenes JPEGs already use rectified/pinhole camera intrinsics. No fabricated distortion coefficients or second undistortion. The source `K` is read from `camera_intrinsic` and saved per frame.
- **Official split:** 700 train + 150 validation scenes in v1.0-trainval; uses the nuScenes devkit's exact 150 validation scene names, not a random 80/20 split.
- **Clips:** 61 RGB frames, 60 relative 9D camera actions; nominal 12Hz video and original microsecond timestamps. One 61-frame clip spans about 5.0 seconds between first and last frames. Default stride 60 frames.

## Output experiments

| Variant | Image pixels / intrinsics | Output FPS | Output folder |
|---|---|---|---|
| `original_native` | Original `CAM_FRONT` pixels (typically 1600×900), original source K | 12 Hz | `.../processed_raw_rawhz/original_native/nuscenes/{train,val}/...` |
| `virtual_native` | 640×384 virtual pinhole, 45° HFOV, same physical extrinsic | 12 Hz | `.../processed_virtual_rawhz/virtual_native/nuscenes/{train,val}/...` |

`original_10hz` is **deliberately rejected** for nuScenes: this experiment asks only for the two native-12Hz variants.

## Installation

From the checkout root (contains `cosmos3prep/`, `cosmos3virtual/`):

```bash
cd /mnt/ufs/users/jaeha-lee/share/jiiyoung/av-feasibility-study
conda activate cosmos

# IMPORTANT: back up if you made custom edits to the previous three_variants.py
cp cosmos3prep/three_variants.py cosmos3prep/three_variants.py.before_nuscenes
unzip -o cosmos3_nuscenes_12hz_two_variants.zip -d .

python -m py_compile cosmos3prep/three_variants.py cosmos3prep/nuscenes_adapter.py
python -m pytest -q tests/test_nuscenes_variants.py
```

Required: existing `numpy`, `opencv-python`, `scipy`, `ffmpeg + libx264` and original cosmos3prep/three_variants dependencies. **No nuScenes SDK or TensorFlow required:** metadata is parsed using the Python standard library and processed in a memory-aware streaming fashion, including large `sample_data.json` / `ego_pose.json`.

## First smoke test (two separate terminals)

**Terminal 1: original K, 12 Hz**

```bash
conda activate cosmos
cd /mnt/ufs/users/jaeha-lee/share/jiiyoung/av-feasibility-study
SMOKE=1 bash scripts/run_nuscenes_original_native.sh
```

**Terminal 2: virtual K, 12 Hz**

```bash
conda activate cosmos
cd /mnt/ufs/users/jaeha-lee/share/jiiyoung/av-feasibility-study
SMOKE=1 bash scripts/run_nuscenes_virtual_native.sh
```

The first metadata scan covers the full metadata (millions of records) even with `--max-clips 1`, because the camera stream is selected by calibration and scene linking. Output is limited to one clip per command.

## Whole dataset (two separate terminals)

```bash
# Terminal 1: original K, native 12 Hz
SMOKE=0 bash scripts/run_nuscenes_original_native.sh

# Terminal 2: virtual K, native 12 Hz
SMOKE=0 bash scripts/run_nuscenes_virtual_native.sh
```

Alternatively, use `python -m cosmos3prep.three_variants prepare` directly:

```bash
# Original K / native 12Hz
export OUT_ROOT=/mnt/ufs/users/jaeha-lee/share/jiiyoung/cosmos3_ji/processed_raw_rawhz
python -m cosmos3prep.three_variants prepare \
  --dataset nuscenes --variant original_native \
  --root /mnt/ufs/users/jaeha-lee/data/finetuned_data/nuscenes_source \
  --camera CAM_FRONT --out-root "$OUT_ROOT" --encoder local --strict

# Common virtual K / native 12Hz
export OUT_ROOT=/mnt/ufs/users/jaeha-lee/share/jiiyoung/cosmos3_ji/processed_virtual_rawhz
python -m cosmos3prep.three_variants prepare \
  --dataset nuscenes --variant virtual_native \
  --root /mnt/ufs/users/jaeha-lee/data/finetuned_data/nuscenes_source \
  --camera CAM_FRONT --out-root "$OUT_ROOT" \
  --virtual-width 640 --virtual-height 384 --virtual-hfov-deg 45 \
  --encoder local --strict
```

### Validation

```bash
python -m cosmos3prep.three_variants validate \
  --out-root /mnt/ufs/users/jaeha-lee/share/jiiyoung/cosmos3_ji/processed_raw_rawhz \
  --variant original_native --only-dataset nuscenes --max-clips 0

python -m cosmos3prep.three_variants validate \
  --out-root /mnt/ufs/users/jaeha-lee/share/jiiyoung/cosmos3_ji/processed_virtual_rawhz \
  --variant virtual_native --only-dataset nuscenes --max-clips 0
```

`--max-clips 0` validates all existing nuScenes clips (may involve decoding many videos). Use `--max-clips 3` for quick checks. `--only-dataset nuscenes` is a new validation filter so it does not decode previously processed KITTI, Waymo and AV2 videos.

### Each clip contains

```text
video.mp4
metadata.json
cosmos_inference.json
K_source_per_frame.npy
K_video_per_frame.npy
distortion_source_per_frame.npy
T_ego_camera.npy
poses_world_ego.npy
poses_local_ego.npy
poses_local_camera.npy
timestamps_s.npy
actions_raw_9d.npy
```

`index.jsonl` is reconstructed at the variant root and includes ALL datasets in that variant directory. Avoid running two jobs that write to the **same** variant root at once (for example two different dataset jobs in `processed_raw_rawhz/original_native`); their `index.jsonl` updates could collide. The two recommended terminals use different variant roots and are safe to run concurrently.

## Critical training/evaluation caveat

**The ZIP implements preprocessing and output validation, not complete 12Hz Cosmos3-Edge training integration.** The current checkout's `cosmos3av` and `cosmos3eval` modules restrict dataset names to KITTI/Waymo/AV2 and/or 10/20Hz; the old `install_native_fps_support.py` likewise accepts only 10/20Hz. Those components must be updated before including nuScenes in Stage-1 or evaluating it end-to-end. No modification to those files is made here.

Even when the source video is 1600×900, Cosmos Framework's `resolution="480"` may resize/pad the input at training time; recorded `K_video` is the *preprocessed MP4 image* calibration, not necessarily the VLM's post-resize camera matrix. For a strict K-conditioning experiment update K for the model-side transform; current Cosmos3-Edge training does not directly ingest K.

### Test scope

`tests/test_nuscenes_variants.py` creates synthetic train + val scenes, camera calibration, poses, 2Hz keyframe and 12Hz sweeps, and checks both outputs with FFmpeg. It does **not** validate live nuScenes data processing or GPU training in the user's environment. Start with `SMOKE=1` then examine the generated metadata before the full 850-scene run.

Official references: https://www.nuscenes.org/data-format and https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/splits.py
