# KITTI + Waymo Ego Motion Baseline — 빠른 실행 가이드

**사용하는 코드 버전**: `ego_motion_kitti_waymo_full` (2026-11-03 재배포).

- `build_index.py`는 **원본 데이터 경로, 시각, 카메라 GT Pose, Extrinsic을 담은 Index JSON만 생성**합니다. KITTI 이미지를 생성하지 않습니다. Waymo에서 `--raw-jpeg-cache`를 지정하면 TensorFlow 없이 학습하기 위한 **원본 JPEG**를 추가로 추출합니다.
- Prepared Mode: `materialize.py`가 공통 Virtual Camera로 변환한 이미지를 **한 번 저장**합니다.
- Online Mode: 학습 시 **원본 이미지에서 즉시 Virtual Camera 변환**을 합니다. Waymo의 원본 JPEG 캐시를 이용할 수 있습니다.
- 모두 공통 Virtual Camera: 640 × 384, 수평 FOV 45°, 동일한 `K_virtual`.
- **중요**: 현재 `train.py`는 **Camera-relative SE(3) Motion Loss**로 학습합니다. Index에 `vehicle_pose`와 `T_vehicle_camera`가 들어 있어도 Vehicle Loss를 수행하지 않습니다. Cosmos 3 6초 전체-영상 모델이 아니라 2-프레임 ResNet18 Baseline입니다.

## 1. 환경

```bash
conda activate scene
python -m pip install -r requirements-train.txt
python -m compileall -q .
PYTHONPATH=. python -m pytest -q tests/
```

PyTorch / Torchvision은 기존 GPU 환경에 맞게 준비하세요. Waymo의 TFRecord 해석은 `waymo-prep` Conda 환경에서 실행합니다.

## 2. KITTI 인덱스 생성

사용자의 이미지와 Pose, 별도 Calibration 경로를 분리해 지정합니다.

```bash
conda activate scene
python build_index.py kitti \
  --root /mnt/ufs/dataset/open-data/KITTI/Kitti-Odometry/dataset \
  --poses-root /mnt/ufs/users/jaeha-lee/share/dataset/kitti/dataset/poses \
  --kitti-calib-root /mnt/ufs/users/jaeha-lee/share/dataset/kitti/dataset/sequences \
  --kitti-frame velodyne \
  --val-seqs 09,10 \
  --out indexes/kitti.json
```

- 기본 Calibration에 `Tr`이 없어도, `--kitti-calib-root`가 가리키는 `00/calib.txt` 등의 `Tr`을 사용합니다.
- **두 Calibration 파일의 `P0`~`P3`가 일치해야 합니다.** 차이가 있으면 오류를 내도록 구성했습니다.
- `--kitti-frame velodyne`은 **LiDAR 기준 대체 프레임(velodyne_proxy)** 이지 실제 차량 중심이 아닙니다.
- 정확한 KITTI 원본 IMU Frame을 원하면 추가 `--kitti-frame imu --kitti-imu-calib-root /path/to/raw_calibration`이 필요합니다.

## 3. Waymo 인덱스 생성

```bash
conda activate waymo-prep
python build_index.py waymo \
  --root /mnt/ufs/dataset/open-data/waymo/perception \
  --max-files-per-split 5 \
  --raw-jpeg-cache cache/waymo_original_jpeg \
  --out indexes/waymo.json
conda activate scene
```

`--max-files-per-split 5`는 디버깅을 위한 소규모 작업입니다. 전체는 `0`입니다. Waymo 왜곡계수는 Manifest에 기록되며 Virtual Camera 변환 시 `cv2.initUndistortRectifyMap`으로 적용합니다. JPEG 캐시는 **Virtual 변환 전 원본 JPEG**입니다.

## 4. 두 Index 검증

```bash
python validate_extrinsics.py --manifest indexes/kitti.json
python validate_extrinsics.py --manifest indexes/waymo.json
```

이 검증은 `T_WC = T_WV @ T_VC` 관계의 **내적 일관성 검사**입니다. 실제 주행 영상과의 물리적 정합성까지 입증하지는 않습니다.

## 5-A. Prepared Mode (Virtual Camera 이미지 사전 저장)

```bash
python materialize.py --index indexes/kitti.json \
  --out-dir processed/kitti --manifest manifests/kitti_prepared.json

python materialize.py --index indexes/waymo.json \
  --out-dir processed/waymo --manifest manifests/waymo_prepared.json

python train.py --mode prepared \
  --kitti manifests/kitti_prepared.json \
  --waymo manifests/waymo_prepared.json \
  --epochs 10 --steps-per-epoch 300 --batch-size 8 --num-workers 2 \
  --out checkpoints/pose_prepared.pt

python evaluate.py --mode prepared \
  --manifest manifests/waymo_prepared.json \
  --checkpoint checkpoints/pose_prepared.pt \
  --split val --max-clips 20
```

같은 경로에 과거 버전 이미지가 있으면 자동 재사용될 수 있습니다. 이전 이미지와 Index의 순서가 불일치할 가능성이 있다면 새로운 `--out-dir`을 쓰거나 `materialize.py --overwrite`를 지정하세요.

## 5-B. Online Mode (학습 시 Virtual Camera 이미지 생성)

```bash
python train.py --mode online \
  --kitti indexes/kitti.json \
  --waymo indexes/waymo.json \
  --epochs 10 --steps-per-epoch 300 --batch-size 8 --num-workers 2 \
  --out checkpoints/pose_online.pt

python evaluate.py --mode online \
  --manifest indexes/waymo.json \
  --checkpoint checkpoints/pose_online.pt \
  --split val --max-clips 20
```

온라인 Waymo에서 `--raw-jpeg-cache`를 사용하지 않았다면 PyTorch 학습 환경에 `waymo_open_dataset.dataset_pb2` 모듈도 필요합니다. TFRecord 랜덤 접근은 느릴 수 있습니다.

## 6. 시각화 (선택)

```bash
python visualize.py --mode prepared \
  --manifest manifests/kitti_prepared.json \
  --split val --frames 61 --camera-height 1.65 \
  --outdir viz/kitti --video

python visualize.py --mode prepared \
  --manifest manifests/waymo_prepared.json \
  --split val --frames 61 \
  --outdir viz/waymo --video
```

MP4 인코딩에는 `ffmpeg`의 H.264 `libx264`가 필요합니다. 지면 투영은 시각화 용도의 근사치이며 Vehicle 중심 GT 오차를 검증하는 수단은 아닙니다.

## 포함 파일

- `build_index.py`, `kitti_extrinsics.py`, `geometry.py`, `camera.py`, `tfrecord_io.py`: 데이터 인덱싱, 좌표계, Virtual Camera
- `materialize.py`, `datasets.py`, `migrate_v2.py`: Prepared / Online 처리
- `model.py`, `train.py`, `evaluate.py`: 학습과 평가
- `inspect_sample.py`, `visualize.py`, `validate_extrinsics.py`: 점검 및 시각화
- `tests/test_pipeline.py`: 소규모 synthetic 검증
- `requirements-*.txt`: 환경 별 의존성

자세한 기술 설명과 제약은 `README.md`를 참고하세요.
