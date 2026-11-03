# 최신 실행 경로 안내

먼저 **[QUICKSTART_KO.md](QUICKSTART_KO.md)** 를 읽으세요. 이 문서에는 최신 `--kitti-calib-root` 경로와 Prepared/Online 명령어가 있습니다. 아래 내용은 상세 기술 설명과 이전 실행 예시를 포함합니다.

# KITTI + Waymo: 두 가지 데이터 로딩 모드의 공통 Ego-Motion Baseline

이 코드는 다음을 분리합니다.

- **공통:** Virtual Camera (`camera.py`), GT camera pose (`geometry.py`), ResNet18 (`model.py`), train/eval.
- **Prepared:** 학습 전 원본 → 640×384 Virtual JPEG 저장 (`materialize.py`), 이후 JPG 읽기.
- **Online:** 원본 PNG 또는 Waymo TFRecord/Raw JPEG를 학습 중 디코딩하고 바로 Virtual Camera로 변환.

두 모드 모두 같은 `K*` (640×384, horizontal FOV 45°), 동일한 Camera Optical Frame(x right/y down/z forward), 동일한 GT를 사용합니다. **Extrinsic은 모델 입력이 아니라 GT 변환에만 사용**됩니다. FPS는 원본 timestamp로 검사하되 별도의 FPS 보간은 하지 않습니다. 6초 전체를 처리하는 Video Transformer가 아니라 **2프레임 → 상대 SE(3) → 61프레임 누적**하는 ResNet18 baseline입니다.

## 0. 디렉터리와 환경

```bash
conda activate scene
python -m pip install -r requirements-train.txt
# torch/torchvision 은 기존 scene 환경의 CUDA용 설치를 사용
python -c 'import torch, torchvision, cv2, scipy; print(torch.__version__)'
```

Waymo 인덱스 생성 또는 TFRecord 직접 접근에는 `from waymo_open_dataset import dataset_pb2`가 필요합니다. 이미 설치한 `waymo-prep` 환경에서 인덱스 및 raw JPEG 추출을 실행할 수 있습니다. 학습 환경 `scene`에는 TensorFlow가 필요하지 않습니다 **(raw JPEG 캐시를 사용하면)**.

KITTI 이미지: `/mnt/ufs/dataset/open-data/KITTI/Kitti-Odometry/dataset`

KITTI Pose: `/mnt/ufs/users/jaeha-lee/share/dataset/kitti/dataset/poses`

Waymo TFRecord: `/mnt/ufs/dataset/open-data/waymo/perception`

위 경로들은 사용자 서버에서 확인된 경로이며, 실행 명령에 반영해 두었습니다.

## 1. 공통 Raw Index 만들기 (두 방식 공통)

먼저 **이미지를 Virtual Camera로 저장하지 않는 작은 메타데이터 인덱스**를 생성합니다.

### KITTI

```bash
conda activate scene
python build_index.py kitti \
  --root /mnt/ufs/dataset/open-data/KITTI/Kitti-Odometry/dataset \
  --poses-root /mnt/ufs/users/jaeha-lee/share/dataset/kitti/dataset/poses \
  --val-seqs 09,10 \
  --out indexes/kitti.json
```

`00~08` Train, `09~10` Val이며, 11~21에는 공개된 GT가 없어 제외됩니다. `times.txt`가 누락되었다면 10Hz를 가정하며 경고합니다.

### Waymo (권장: Raw JPEG 캐시 포함)

```bash
conda activate waymo-prep
python build_index.py waymo \
  --root /mnt/ufs/dataset/open-data/waymo/perception \
  --max-files-per-split 5 \
  --raw-jpeg-cache cache/waymo_original_jpeg \
  --out indexes/waymo.json
conda activate scene
```

`training`은 Train, `validation`은 Val. 이때 `cache/waymo_original_jpeg`의 JPEG는 **원본**이며 Virtual Camera로 변환되지 않았습니다. 원본 JPEG 캐시는 PyTorch 학습 환경에 Waymo 패키지가 없어도 Online Mode를 실행하려는 실용적인 중간 단계입니다. **Raw JPEG 캐시가 있으면 Online Mode는 Virtual 이미지를 저장하지 않습니다.**

완전 무캐시(TFRecord 직접 Online) 방식을 원하면 `--raw-jpeg-cache`를 생략하세요. 이 경우 `scene`에서 `waymo_open_dataset.dataset_pb2`를 import할 수 있어야 합니다. 설치 과정에서 `scene`의 TensorFlow/PyTorch 의존성이 충돌할 수 있으므로 기존 환경에 무리하게 설치하지 마세요. **기술적으로 가능한 방식이지만 TFRecord 랜덤 접근은 디스크 I/O 비용이 상당히 큽니다.**

`--max-files-per-split 5`는 빠른 테스트용입니다. 전체 데이터는 `0`입니다. Waymo GT는 `CameraImage.pose`와 `pose_timestamp`를 사용합니다. 결측 또는 timestamp 갭은 연결하지 않습니다.

### 공통 생성물

```text
indexes/kitti.json             # 이미지 경로, Camera GT, 원본 K, timestamps
indexes/waymo.json             # TFRecord byte offset (및 선택적 Raw JPEG 경로), GT, 원본 K
```

## 2-A. Prepared Mode (미리 Virtual 이미지 저장)

```bash
conda activate scene
python materialize.py --index indexes/kitti.json \
  --out-dir processed/kitti --manifest manifests/kitti_prepared.json

python materialize.py --index indexes/waymo.json \
  --out-dir processed/waymo --manifest manifests/waymo_prepared.json

python inspect_sample.py --manifest manifests/kitti_prepared.json \
  --mode prepared --split train --out viz/kitti_prepared.jpg

python inspect_sample.py --manifest manifests/waymo_prepared.json \
  --mode prepared --split val --out viz/waymo_prepared.jpg

python train.py --mode prepared \
  --kitti manifests/kitti_prepared.json \
  --waymo manifests/waymo_prepared.json \
  --epochs 10 --steps-per-epoch 300 --batch-size 8 --num-workers 2 \
  --out checkpoints/pose_prepared.pt

python evaluate.py --mode prepared \
  --manifest manifests/waymo_prepared.json \
  --checkpoint checkpoints/pose_prepared.pt --max-clips 20
```

흐름:

```text
KITTI PNG -----------┐
                     ├─ build_index → raw manifest
Waymo TFRecord ------┘         ↓
                            materialize → Virtual JPG + prepared manifest
                                                  ↓
                               DataLoader reads JPG + GT + constant K*
                                                  ↓
                                      ResNet18 → relative SE(3)
                                                  ↓
                                          loss / RRE RTE ATE
```

## 2-B. Online Mode (학습에서 Virtual 변환)

```bash
conda activate scene
python inspect_sample.py --manifest indexes/kitti.json \
  --mode online --split train --out viz/kitti_online.jpg
python inspect_sample.py --manifest indexes/waymo.json \
  --mode online --split val --out viz/waymo_online.jpg

python train.py --mode online \
  --kitti indexes/kitti.json \
  --waymo indexes/waymo.json \
  --epochs 10 --steps-per-epoch 300 --batch-size 8 --num-workers 2 \
  --out checkpoints/pose_online.pt

python evaluate.py --mode online \
  --manifest indexes/waymo.json \
  --checkpoint checkpoints/pose_online.pt --max-clips 20
```

흐름:

```text
KITTI PNG -- read -------┐
                         ├─ DataLoader ─ warp using K_src,D → virtual RGB
Waymo Raw JPEG -- read --┘                       |
                            or Waymo TFRecord ---┘
                                                ↓
                               virtual RGB + constant K* + Camera GT
                                                ↓
                                      ResNet18 → relative SE(3)
                                                ↓
                                          loss / RRE RTE ATE
```

Online은 **virtual JPG를 별도 저장하지 않으며** `cv2.remap`을 호출하는 시점만 Prepared와 다릅니다. `--raw-jpeg-cache`로 추출한 원본 JPEG는 여전히 디스크를 사용합니다. Waymo 직접 TFRecord 모드는 이 원본 JPEG 캐시도 불필요합니다.

## 3. Manifest v3 구조

```json
{
  "format_version": 3,
  "stage": "raw",
  "camera": {"K_virtual": [[772.548,0,320],[0,772.548,192],[0,0,1]],
             "virtual_size": [640,384], "virtual_hfov_deg": 45.0},
  "sequences": [{
    "name": "kitti_00_0", "source": "kitti", "split": "train",
    "K_src": [[700,0,600],[0,700,190],[0,0,1]],
    "distortion": [0,0,0,0,0],
    "frames": [{"timestamp_s": 0.0, "raw_path": "/.../000000.png",
                "pose": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}]
  }]
}
```

`K_src`, `distortion`은 영상 변환에 사용되고, `K_virtual`은 학습 모델에 전달됩니다. `pose`는 camera optical → world의 **Global Pose**입니다. 인접 프레임에서 `inv(T_t) @ T_{t+1}`을 계산한 결과가 학습 GT입니다. `prepare` 단계에서 이미 Virtual K에 맞춘 이미지를 저장하면 DataLoader에서 다시 워핑하지 않습니다.

Waymo Raw Mode의 각 frame에는 `tfrecord`, `offset`이 들어 있고, `--raw-jpeg-cache` 사용 시 `raw_path`도 들어 있습니다. Prepared Mode에서는 `prepared_path`가 추가됩니다.

## 4. 기존 v2 이미지 재사용 (선택)

이미 기존 `prepare_kitti.py`, `prepare_waymo.py`로 만든 Virtual JPG가 있다면:

```bash
python migrate_v2.py --input processed/kitti/kitti_manifest.json \
  --out manifests/kitti_legacy_prepared.json
python migrate_v2.py --input processed/waymo/waymo_manifest.json \
  --out manifests/waymo_legacy_prepared.json
```

위 명령은 **Virtual 이미지를 다시 생성하지 않습니다**. 단, 이전 코드가 생성한 Pose의 Calibration 정확성, Timestamp 갭, 데이터 누락 여부는 자동 수정하지 않으므로, 논문 실험용으로는 v3 Raw Index를 새로 만드는 것을 추천합니다. 레거시 manifest에는 원본 영상 파일 위치, 카메라 Intrinsic이 없으므로 Online Mode에 사용할 수 없습니다.

## 5. 평가 정의 / 제한

- 학습 출력: `[tx, ty, tz, rotvec_x, rotvec_y, rotvec_z]`; `rotvec` 단위 rad, Translation m.
- RRE: 인접 프레임 Relative Rotation의 SO(3) geodesic angle, 각도(degree).
- RTE: 인접 프레임 Relative Translation L2 오차(m).
- ATE: 6초 61프레임에서 Relative SE(3)를 적분한 후, **첫 Pose에서 정렬하고 별도 Scale / SE(3) fitting 없이** Translation RMSE(m).
- Camera-relative action을 예측/평가하며 **Vehicle 좌표로 변환하지 않습니다**. 필요하다면 Extrinsic으로 평가 단계에서 변환할 수 있습니다.
- Virtual Camera는 **intrinsic만 통일**, 카메라 물리적 높이·방향(Extrinsic)은 통일하지 않습니다.
- Waymo Camera Image는 rolling shutter로 인한 잔여 왜곡이 있을 수 있으며 GT 캡처 시점은 `image.pose_timestamp`에 맞춥니다.
- VLA/Cosmos 3의 논문과 모델/데이터/학습 프로토콜이 다르므로 표 18 수치의 재현으로 해석하지 마세요.
- **공통 K*를 항상 입력하므로 K Encoder는 상수 특징만 학습합니다.** K 조건부 일반화 효과를 검증하려면 카메라 Intrinsic Augmentation이 필요합니다.

## 6. 신뢰성 검증

```bash
python -m compileall -q .
python -m pytest -q tests/
```

검증 스크립트가 정상이라면 별도의 Waymo 라이브러리 없이 KITTI의 공통 이미지 전처리, SE(3) 변환, PairDataset, Materialization 및 ResNet18 Forward까지 확인합니다. **실제 서버의 KITTI/Waymo 전체 데이터 실행은 별도 검증이 필요합니다.**

## 7. 이전과 같은 BEV + 실제 이미지 Road Projection + H.264 영상

새 Manifest v3에서 `visualize.py`를 실행합니다. Prepared/Online 둘 다 지원합니다.

```bash
python visualize.py --mode prepared \
  --manifest manifests/kitti_prepared.json --split val \
  --sequence-index 0 --frames 61 --camera-height 1.65 \
  --outdir viz/kitti --video

python visualize.py --mode online \
  --manifest indexes/waymo.json --split val \
  --sequence-index 0 --frames 61 --outdir viz/waymo --video
```

- `projection_on_road.png`: 실제 Virtual RGB 첫 프레임에 GT 지면 경로를 투영
- `bev.png`: 같은 궤적의 카메라 상대 Top-down 그래프
- `projection_video.mp4`: H.264 / yuv420p, 브라우저 재생 가능

KITTI는 카메라 높이 1.65m와 평탄 지면의 근사치입니다. Waymo v3 인덱스는 Camera Extrinsic과 각 시점 Vehicle Pose까지 별도로 보유해, 카메라 설치 지점 아래의 **vehicle z=0 참조점**을 투영합니다(실측 도로면은 아님). 직접 TFRecord를 열 필요 없이 캐시된 원본 JPEG만으로도 시각화 가능합니다.

---

## 8. Extrinsic 포함 Manifest 생성 (추가 기능)

**왜 추가하는가?** Camera Motion을 추정한 뒤 Vehicle-relative Action으로 옮기려면
`T_vehicle_camera`가 있어야 한다. 여기서 행렬명은 `T_vehicle_camera`지만,
**실제 기준 프레임이 무엇인지는 반드시 `vehicle_frame`으로 확인해야 한다.**
모델 입력, loss, evaluation은 이번 버전에서도 기존처럼 Camera-relative Motion이다.
이번 수정은 정확한 변환 정보를 Manifest에서 재사용할 수 있게 하는 것이다.

### KITTI: 기존 Odometry Calibration만 있을 때 (별도 Raw 다운로드 필요 없음)

```bash
conda activate scene
python build_index.py kitti \
  --root /mnt/ufs/dataset/open-data/KITTI/Kitti-Odometry/dataset \
  --poses-root /mnt/ufs/users/jaeha-lee/share/dataset/kitti/dataset/poses \
  --kitti-frame velodyne \
  --val-seqs 09,10 \
  --out indexes/kitti.json
```

필수 파일:
- `sequences/XX/calib.txt`: `P0`, `P2`, `Tr` (Velodyne → Camera 0)
- `sequences/XX/image_2/*.png`
- `poses/XX.txt` (별도 `--poses-root` 경로 허용)

저장 내용: `vehicle_frame=velodyne_proxy`, `T_vehicle_camera` = **Cam2 optical → Velodyne**, 
`vehicle_pose` = Velodyne 기준의 Global Pose. **이는 실제 차량 중심/후륜축 원점이 아니다.**
따라서 Waymo Vehicle Frame과 동일 원점의 GT라고 간주하면 안 된다.

### KITTI: IMU 기준으로 만들기 (추가 Raw Calibration 준비 시)

해당 Sequence의 Raw 녹화일자에 대응하는 `calib_imu_to_velo.txt`를 다음처럼 보관한다.

```text
/mnt/ufs/dataset/open-data/KITTI/raw-calib/
  2011_10_03/calib_imu_to_velo.txt
  2011_09_26/calib_imu_to_velo.txt
  2011_09_30/calib_imu_to_velo.txt
```

이 RAW Calibration은 odometry 이미지 전체를 다운로드할 필요 없이 KITTI Raw Data 사이트에서 날짜별 Calibration만 확보하면 된다.

```bash
python build_index.py kitti \
  --root /mnt/ufs/dataset/open-data/KITTI/Kitti-Odometry/dataset \
  --poses-root /mnt/ufs/users/jaeha-lee/share/dataset/kitti/dataset/poses \
  --kitti-frame imu \
  --kitti-imu-calib-root /mnt/ufs/dataset/open-data/KITTI/raw-calib \
  --out indexes/kitti_imu.json
```

이 경우 `vehicle_frame=imu`, `T_vehicle_camera` = Cam2 optical → IMU, `vehicle_pose` = IMU 기준 Global Pose.
KITTI IMU도 자차 후륜축 중심과 같은 것은 아니며 Waymo와 기준 원점 차이를 감안해야 한다.

### Waymo: 카메라 Optical Frame과 일치하는 Extrinsic 추가

```bash
conda activate waymo-prep
python build_index.py waymo \
  --root /mnt/ufs/dataset/open-data/waymo/perception \
  --max-files-per-split 5 \
  --raw-jpeg-cache cache/waymo_original_jpeg \
  --out indexes/waymo.json
conda activate scene
```

- `T_vehicle_camera_raw`: Waymo 카메라 원래 좌표축 → Waymo Vehicle
- `T_vehicle_camera`: **Optical Camera → Waymo Vehicle** (모델의 Camera GT와 동일 프레임)
- `vehicle_frame=waymo_vehicle`, `vehicle_pose`는 촬영 시점 차량 GT

### Manifest 추가 필드

```json
{
  "name": "kitti_00_0",
  "source": "kitti",
  "vehicle_frame": "velodyne_proxy",
  "vehicle_frame_note": "LiDAR origin, not physical ego center",
  "extrinsic_source": "odometry/calib.txt:Tr,P0,P2",
  "T_vehicle_camera": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
  "frames": [
    {"pose": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
     "vehicle_pose": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
  ]
}
```

예시는 JSON 필드 형태만 표시한 것이며 **행렬값은 실제 Calibration에서 계산한 값을 사용한다.**
`pose`는 여전히 Camera Optical → World이며, 카메라 상대 모션 학습에 사용된다.

### Prepared Mode: 기존 Virtual JPG 재사용

먼저 위 `build_index.py`를 **다시 실행**해 Extrinsic을 포함한 `indexes/*.json`을 만든다.
그다음 `materialize.py`를 실행한다. `--overwrite`를 지정하지 않으면
**이미 존재하는 640×384 Virtual JPEG는 덮어쓰지 않고**, 새 `prepared_manifest.json`에
추가 메타데이터만 반영한다. 기존 저장 폴더 경로와 순서가 같다는 전제이며,
Virtual Camera 설정(45°, 640×384)을 바꾼 적이 있다면 재생성해야 한다.

```bash
python materialize.py --index indexes/kitti.json \
  --out-dir processed/kitti --manifest manifests/kitti_prepared.json
python materialize.py --index indexes/waymo.json \
  --out-dir processed/waymo --manifest manifests/waymo_prepared.json
```

### 검증

```bash
python validate_extrinsics.py --manifest indexes/kitti.json
python validate_extrinsics.py --manifest indexes/waymo.json
python -m pytest -q tests/
```

정상이라면 `T_WC ≈ T_WV @ T_VC`가 두 데이터셋에서 성립한다.
이 테스트는 **계산된 변환끼리의 수학적 일관성**을 확인하며,
원본 센서 캘리브레이션의 정확도까지 입증하지는 않는다.

### Waymo 왜곡 보정 적용 위치

- `build_index.py`: `cam.intrinsic`에서 `[fx,fy,cx,cy,k1,k2,p1,p2,k3]`를 읽어
  `K_src`, `distortion` 저장
- `camera.py`: `cv2.initUndistortRectifyMap(K_src, distortion, R=I, newCameraMatrix=K_virtual, ...)`
  → `cv2.remap`으로 유효 픽셀 검사 및 왜곡 제거/Virtual Warp 수행
- `materialize.py`: Prepared Mode에서 위 보정 결과를 JPEG로 저장
- `datasets.py`: Online Mode에서 매번 위 보정을 실행

KITTI `image_2`는 이미 rectified 이미지이므로 `distortion=[0,0,0,0,0]`을 사용한다.
Waymo의 방사/접선 왜곡 계수는 OpenCV 포맷과 일치하지만, rolling-shutter 모션 왜곡은 별도 문제다.

## KITTI `calib.txt`에 `P0`~`P3`만 있는 경우 (2026-11 수정)

이번 버전 `build_index.py`에 `--kitti-calib-root` 옵션을 추가했습니다. 원본 이미지 디렉터리의 calibration은 수정하지 않습니다. 원본에 `Tr`이 없다면 **공식 KITTI Odometry calibration ZIP에서 해당 sequence의 `Tr`을 읽고, 원본 `P0`~`P3`와 값이 맞는지 검사한 후** Extrinsic을 계산합니다. 값이 맞지 않으면 명시적으로 오류를 발생시킵니다. 빈 `Tr`, 임의의 항등행렬로 대체하지 않습니다.

```bash
cd /mnt/ufs/dataset/open-data/KITTI
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip
mkdir -p odometry_calib_extra
unzip -n data_odometry_calib.zip -d odometry_calib_extra

# 먼저 zip 내부 경로 확인
unzip -l data_odometry_calib.zip | head -30

python build_index.py kitti \
  --root /mnt/ufs/dataset/open-data/KITTI/Kitti-Odometry/dataset \
  --poses-root /mnt/ufs/users/jaeha-lee/share/dataset/kitti/dataset/poses \
  --kitti-calib-root /mnt/ufs/dataset/open-data/KITTI/odometry_calib_extra \
  --kitti-frame velodyne \
  --val-seqs 09,10 \
  --out indexes/kitti.json
```

참고: `--kitti-frame velodyne`의 'vehicle'은 실제 자차 중심이 아니라 **LiDAR 기준 proxy**입니다. Camera-only VO/IDM 학습은 `P0`~`P3`만으로도 가능하지만 이 버전의 vehicle-enabled 인덱서는 `Tr`을 요구합니다.
