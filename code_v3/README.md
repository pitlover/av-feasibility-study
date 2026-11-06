# Cosmos3-Edge AV Inverse Dynamics — 최종 통합 코드

> **기준**: NVIDIA `cosmos-framework` git revision `98c35eda`와 사용자의 KITTI 시험 학습 / Export / Inverse Dynamics 추론 / RRE·RTE·ATE 평가 성공 로그에서 사용한 코드 조합.
>
> **검증 범위**: 제공된 과거 코드 스냅샷을 통합하고, CPU 단위 테스트 및 FFmpeg H.264 렌더 테스트를 통과시켰습니다. **네 서버에서 실제 성공한 파일의 마지막 수정분을 직접 읽은 것은 아니므로** 이 배포본 전체를 GPU에서 새로 검증했다고 주장하지 않습니다. Waymo·AV2 전처리, AV2 추가 학습, 원본 Edge와의 정확한 action normalization은 별도 실데이터 검증이 필요합니다.

## 1. 구성

```text
cosmos3_ji/
  cosmos3prep/           # KITTI, Waymo v1, AV2 원본 -> 61-frame video + 60x9 GT
  cosmos3av/             # Edge Inverse Dynamics 커스텀 학습 (dataset selection)
  cosmos3eval/           # prepare, export, infer, score, compare, render
  check_config_contract.py
  scripts/
    env.sh              # 공통 BASE/FRAMEWORK/PROCESSED 경로
    evaluate.sh         # 모델 이름을 인수로 받고 Infer/Score/Render 수행
  tests/
  requirements-*.txt
  README.md

  cosmos-framework/      # 별도 설치된 NVIDIA 공식 Framework — ZIP에 포함되지 않음
  processed/             # 기존 전처리 데이터 — ZIP에 포함되지 않음
  checkpoints/           # 기존 모델 — ZIP에 포함되지 않음
  runs/                  # 학습 체크포인트 — ZIP에 포함되지 않음
  exports/               # Export된 모델 — ZIP에 포함되지 않음
  evaluation/            # 평가 결과 — ZIP에 포함되지 않음
```

### 어떤 수정이 반영됐나

- Nano/Qwen 구조 대신 **Cosmos3-Edge / Nemotron3** 모델 설정. (학습기의 optimizer/packing 기본값 일부는 기존 LIBERO Nano SFT 레시피에서 가져온 커스텀 구성입니다. **공식 검증된 AV SFT 레시피는 아닙니다.**)
- `cosmos3av/cli.py`의 `experiment = "action_inverse_dynamics_av_edge"` 설정 및 안전 검사.
- Training과 Export에서 AV Dataset Metadata (`60 action steps`, `10 fps`, `av`) 공유.
- KITTI 이미지·GT Pose·Calibration 각각 다른 경로 지원, KITTI native timing, H.264 MP4 생성.
- Waymo Perception v1 TFRecord 및 AV2 Sensor 20Hz→10Hz 로더.
- Export 실행 시 **원본 `config.yaml`을 보존**하고 `_type`, PyTorch dtype 문자열을 수정한 Export 전용 설정 파일을 자동 생성/검증.
- 공식 `sample_outputs.json` → `outputs[0].content.action` 파서 수정.
- RRE / RTE / ATE 및 BEV 정량 결과, **GT/Pred H.264 정성 평가 영상(render)**.

## 2. 기존 서버에 적용

**백업 후** 압축 파일을 `cosmos3_ji` 폴더에 업로드해 압축을 풉니다. `processed/`, `runs/`, `exports/`, `evaluation/`은 지우지 마세요.

```bash
cd /mnt/ufs/users/jaeha-lee/share/jiiyoung/cosmos3_ji
unzip cosmos3_edge_final_bundle.zip

# 기존 코드 보존
cp -a cosmos3prep "cosmos3prep.backup_$(date +%Y%m%d_%H%M%S)" 2>/dev/null || true
cp -a cosmos3av "cosmos3av.backup_$(date +%Y%m%d_%H%M%S)" 2>/dev/null || true
cp -a cosmos3eval "cosmos3eval.backup_$(date +%Y%m%d_%H%M%S)" 2>/dev/null || true
cp -a check_config_contract.py check_config_contract.py.backup 2>/dev/null || true

cp -a cosmos3_edge_final_bundle/cosmos3prep/. cosmos3prep/
cp -a cosmos3_edge_final_bundle/cosmos3av/. cosmos3av/
cp -a cosmos3_edge_final_bundle/cosmos3eval/. cosmos3eval/
cp cosmos3_edge_final_bundle/check_config_contract.py ./
cp -a cosmos3_edge_final_bundle/scripts ./

source scripts/env.sh
```

※ Bash의 `source` 명령은 현재 터미널에만 적용됩니다. 새 터미널에서 다시 실행하세요. 설치 전 현재 `cosmos3av/experiment.py`가 정상 실행 중이라면 우선 ZIP을 별도 디렉터리에 풀어 diff 비교하는 것도 안전합니다.

## 3. 원본 데이터 전처리 (필요할 때만)

기존 `processed/`가 모두 있으면 이 단계는 생략하세요. 패키지를 다시 받아도 전처리를 재실행할 필요 없습니다.

**KITTI (팀 공용 이미지 + 개인 Pose/Calibration)**

```bash
python -m cosmos3prep.cli kitti \
  --root /mnt/ufs/dataset/open-data/KITTI/Kitti-Odometry/dataset \
  --poses-dir /mnt/ufs/users/jaeha-lee/share/dataset/kitti/data_odometry_poses/dataset/poses \
  --calib-root /mnt/ufs/users/jaeha-lee/share/dataset/kitti/dataset \
  --sequences 00 01 02 03 04 05 06 07 08 09 10 \
  --camera image_2 --kitti-timing native --stride 60 \
  --out "$PROCESSED" --video-codec h264 --encoder local --strict
```

※ KITTI 원본은 nominal 10Hz이고 native 모드에서 원본 시간 간격을 보존합니다. MP4 FPS만 정확한 10으로 고정됩니다.

**Waymo Perception v1 TFRecord** (TensorFlow와 `waymo_open_dataset`이 설치된 Python 3.10 환경):

```bash
/home/ad-group/miniconda3/envs/waymo-prep/bin/python -m cosmos3prep.cli waymo \
  --root /mnt/ufs/dataset/open-data/waymo/perception/training \
  --camera FRONT --out "$PROCESSED" --stride 60 \
  --video-codec h264 --encoder local --strict
```

`--root`는 실제 TFRecord 폴더를 지정하세요. 1개 샘플 테스트는 `--max-logs 1 --max-clips 1`을 추가하면 됩니다.

**Argoverse2 Sensor** (경로는 실제 파일 구조에 맞게 수정):

```bash
python -m cosmos3prep.cli av2 \
  --root /path/to/av2/sensor/train \
  --camera stereo_front_left --out "$PROCESSED" --stride 60 \
  --video-codec h264 --encoder local --strict
```

Val은 `--root /path/to/av2/sensor/val`로 별도로 처리합니다. AV2 Post-training에는 `av2/train` clip이 반드시 있어야 합니다.

검증:

```bash
python -m cosmos3prep.cli validate --out "$PROCESSED" --max-clips 5
```

## 4. Cosmos3-Edge 학습 (Stage 1)

**사전 준비**: NVIDIA 공식 `cosmos-framework` 코드 및 훈련 환경, `Cosmos3-Edge` DCP 초기 가중치, Wan2.2 VAE. 이 세 가지 대용량 파일은 ZIP에 넣지 않았습니다.

```bash
source scripts/env.sh
export EDGE_DCP="/실제/경로/Cosmos3-Edge"     # 초기 Edge DCP 디렉터리
export VAE="/실제/경로/Wan2.2_VAE.pth"       # Wan2.2 VAE 파일

# KITTI만, 전체 336 clips 사용 예시 (실제 개수에 맞추세요)
python -m cosmos3av.cli plan \
  --stage pretrain --datasets kitti \
  --processed-root "$PROCESSED" \
  --init-checkpoint "$EDGE_DCP" --wan-vae "$VAE" \
  --outdir "$BASE/runs/stage1_kitti_300" \
  --epoch-clips 336 --max-iters 300 --save-every 50 \
  --gpus 4 --full-preflight

python -m cosmos3av.cli launch \
  --run-dir "$BASE/runs/stage1_kitti_300" \
  --framework-root "$FRAMEWORK" --execute
```

**데이터셋 선택**: `--datasets waymo`, `--datasets kitti waymo --weights kitti:1,waymo:1` 모두 지원합니다. `--epoch-clips`는 고유 데이터 개수가 아니라 생성할 학습 manifest **샘플 엔트리 수**입니다. `--max-iters`만큼 최적화 스텝을 반복합니다.

이미 성공한 10-step 모델을 다시 만들 필요는 없습니다. 기존 `runs/stage1_kitti_exporttest/`를 사용할 수 있습니다.

설정 검사(기존 `run.json` 사용):

```bash
export RUN="$BASE/runs/stage1_kitti_exporttest"
eval "$(python - <<'PY'
import json,os,shlex
p=os.environ['RUN']+'/run.json'; d=json.load(open(p))
print('export BASE_CHECKPOINT_PATH='+shlex.quote(d['init_checkpoint']))
print('export WAN_VAE_PATH='+shlex.quote(d['wan_vae']))
PY
)"
export COSMOS_AV_MANIFEST="$RUN/train.jsonl"
python check_config_contract.py --sft-toml "$RUN/sft_config.toml"
```

## 5. DCP → 추론용 Export (기존 학습 Weight 재사용)

```bash
source scripts/env.sh
export MODEL_NAME="stage1_kitti_exporttest"  # export 폴더 이름. 슬래시(/)를 넣지 마세요.
export TRAIN_RUN="$BASE/runs/$MODEL_NAME/outputs/cosmos3_av_id/pretrain/$MODEL_NAME"
export DCP="$TRAIN_RUN/checkpoints/iter_000000010"

python -m cosmos3eval.cli export \
  --framework-root "$FRAMEWORK" --dcp "$DCP" \
  --config "$TRAIN_RUN/config.yaml" \
  --out "$BASE/exports/$MODEL_NAME" --execute
```

Export 전 처리는 `cosmos3eval/export_compat.py`가 수행합니다. 원본 `config.yaml`과 DCP는 변경하지 않고 별도의 `exports/<name>.export_config.yaml`을 생성합니다.

다른 iteration의 Weight를 Export할 때는 `DCP`와 `MODEL_NAME`을 각각 해당 경로로 바꾸세요. 체크포인트를 다른 이름으로 Export할 경우 `TRAIN_RUN`은 **실제 학습 실행 이름**을 가리켜야 합니다.

## 6. Evaluation 준비 → 추론 → 점수 → 정성 평가 영상

**Prepare는 평가 영상과 GT를 골라 selection.jsonl을 만드는 과정**이며 한 평가 세트당 한 번만 하면 됩니다.

```bash
source scripts/env.sh
python -m cosmos3eval.cli prepare \
  --processed-root "$PROCESSED" --datasets kitti \
  --split val --max-clips-per-dataset 1 \
  --out "$BASE/evaluation/kitti_val_1"
```

이미 `selection.jsonl`이 있다면 Prepare를 생략하고 기존 평가 세트를 재사용하세요. 같은 모델 비교에는 동일한 평가 세트를 사용해야 합니다. **새 평가 세트를 선택할 때만** `--force`를 사용하세요.

**한 명령으로 Inference + Score + GT/Pred 정성 영상**:

```bash
# 이전 TorchCodec/NPP 심볼 오류가 있었던 머신에서만 활성화
export COSMOS_PRELOAD_NPP=1
bash scripts/evaluate.sh stage1_kitti_exporttest kitti_val_1 all
```

단계별 실행을 원하면:

```bash
bash scripts/evaluate.sh stage1_kitti_exporttest kitti_val_1 infer
bash scripts/evaluate.sh stage1_kitti_exporttest kitti_val_1 score
bash scripts/evaluate.sh stage1_kitti_exporttest kitti_val_1 render
```

`evaluate.sh`는 `MODEL_NAME` 누락, 슬래시 포함, Export 모델 폴더·평가 선택 목록 누락을 미리 검사합니다. GPU 모델 추론은 `infer` 또는 `all`에서만 실행됩니다. 이미 추론 결과가 존재하면 `score`, `render`만 실행하면 됩니다.

**정성 평가 영상만 별도 실행**:

```bash
python -m cosmos3eval.cli render \
  --eval-dir "$BASE/evaluation/kitti_val_1" \
  --pred-dir "$BASE/evaluation/pred_stage1_kitti_exporttest" \
  --out-dir "$BASE/evaluation/pred_stage1_kitti_exporttest/qualitative" \
  --max-clips 1
```

정성 영상은 원본 RGB 프레임 우상단에 **GT(녹색) vs Pred(빨간색) 로컬 BEV 궤적**을 넣습니다. 이는 카메라 영상의 실제 노면 좌표로 투영한 영상이 아닙니다. 렌더링과 점수 계산은 **같은 SE(3) decoder**를 사용하며 H.264로 저장됩니다.

### Evaluation 결과

```text
evaluation/pred_<MODEL_NAME>/
  <clip_id>/sample_outputs.json
  metrics/
    summary.json        # RRE, RTE, ATE
    per_clip.csv
    per_frame.csv
    bev_<clip_id>.png
    trajectories/<clip_id>_pred.npy
  qualitative/
    <clip_id>_gt_pred.mp4
```

여러 모델의 `summary.json`을 비교할 때는:

```bash
python -m cosmos3eval.cli compare \
  --summaries "$BASE/evaluation/pred_stage1_kitti_exporttest/metrics/summary.json" \
              "$BASE/evaluation/pred_ANOTHER_MODEL/metrics/summary.json" \
  --out "$BASE/evaluation/model_compare.csv"
```

## 7. Stage 2 — Argoverse2 추가 학습

AV2 **train clip**이 준비된 이후에만 실행합니다. Stage 1의 선택된 DCP를 `--init-checkpoint`에 지정합니다.

```bash
python -m cosmos3av.cli plan \
  --stage posttrain --datasets av2 \
  --processed-root "$PROCESSED" \
  --init-checkpoint /path/to/stage1/checkpoints/iter_000000300 \
  --wan-vae "$VAE" --outdir "$BASE/runs/stage2_av2" \
  --epoch-clips 500 --max-iters 200 --save-every 50 --gpus 4 --full-preflight
python -m cosmos3av.cli launch \
  --run-dir "$BASE/runs/stage2_av2" --framework-root "$FRAMEWORK" --execute
```

## 8. 검증 및 주의

코드 자체 검증(별도의 Python 프로세스로 실행하여 NVIDIA 모듈 Mock 충돌 방지):

```bash
python -m pytest -q tests/prep
python -m pytest -q tests/eval
python -m pytest -q tests/train/test_pipeline.py tests/train/test_cli_guard.py
python -m pytest -q tests/train/test_experiment_contract.py
```

**중요한 해석 한계**: 현재 RRE/RTE/ATE는 **예측 Action이 전처리 GT와 같은 9D rotation encoding, `backward_framewise` convention, meter scale, action normalization을 따른다는 가정**에서 계산됩니다. GT Action→Pose roundtrip은 검사하지만, **원본 Cosmos3-Edge 체크포인트의 출력 정규화·좌표계 의미가 GT와 일치한다는 독립적 검증은 아직 남아 있습니다.** 숫자가 나와도 성능 비교 결론에 앞서 이 부분을 반드시 확인하세요. 기존 KITTI 10-step 단일 clip의 8.9367°/0.4683m/25.5100m는 파이프라인 동작 검증 수치입니다.

장치 의존 사항: PyTorch/CUDA/TorchCodec/Wan VAE/FFmpeg를 바꿀 때는 기존에 학습과 Export에 성공한 Conda 환경의 버전 충돌을 주의하세요. Waymo 전처리는 별도 `waymo-prep` Python 환경으로 실행하는 것을 권장합니다.
