# Cosmos3-Edge AV Inverse Dynamics — experimental Stage 1 / Stage 2

이 패키지는 이전 Nano 실험 코드를 **Edge 아키텍처**로 바꾼 연구용 시도입니다. 공식 NVIDIA AV inverse-dynamics training recipe는 아니며, **GPU forward/backward 실행 및 checkpoint 호환성은 이 환경에서 검증되지 않았습니다**. `--execute`를 넣기 전에 아래의 점검을 완료해야 합니다.

## 위치
- 팀 데이터의 전처리 결과 `processed/index.jsonl`은 그대로 사용합니다.
- `cosmos3av/` 폴더가 포함된 이 디렉터리에서 `python -m cosmos3av.cli ...` 실행합니다.
- 실제 학습 `launch`는 실행 시 공식 `cosmos-framework/` 저장소를 subprocess의 작업 디렉터리로 사용합니다. 공식 저장소에 코드를 복사하지 않아도 됩니다.
- Edge DCP, Wan2.2 VAE, `cosmos-framework` 코드를 사전에 준비해야 합니다.

## Step 5A — Stage 1 설정 (KITTI + Waymo 모두 사용)
```bash
cd /mnt/ufs/users/jaeha-lee/share/jiiyoung/cosmos3_ji/cosmos3_edge_av_stage1
export BASE=/mnt/ufs/users/jaeha-lee/share/jiiyoung/cosmos3_ji
export PROCESSED="$BASE/processed"
export EDGE_DCP="$BASE/checkpoints/Cosmos3-Edge"
export VAE=/실제/Wan2.2_VAE.pth/경로
export FRAMEWORK="$BASE/cosmos-framework"

python -m cosmos3av.cli plan \
  --stage pretrain --datasets kitti waymo --weights kitti:1,waymo:1 \
  --processed-root "$PROCESSED" \
  --init-checkpoint "$EDGE_DCP" --wan-vae "$VAE" \
  --outdir ./runs/stage1_joint_smoke --epoch-clips 100 \
  --max-iters 10 --save-every 10 --gpus 4 \
  --full-preflight --check-video
```
`--datasets kitti` 또는 `--datasets waymo`로 한 데이터셋만 선택할 수 있습니다. `--gpus`는 실제 GPU 개수로 바꿉니다.

## Step 5B — 확인 및 소규모 실행
```bash
python -m cosmos3av.cli smoke --manifest ./runs/stage1_joint_smoke/train.jsonl
python -m cosmos3av.cli launch --run-dir ./runs/stage1_joint_smoke --framework-root "$FRAMEWORK"
# 데이터 동기화, pretrained action normalization 및 GPU 용량을 검증한 뒤:
python -m cosmos3av.cli launch --run-dir ./runs/stage1_joint_smoke --framework-root "$FRAMEWORK" --execute
```

**중요한 선행 확인**
1. `EDGE_DCP`가 실제 Edge용 DCP인지, pretrained AV action head를 로드했는지 (missing/unexpected keys 확인).
2. `actions_raw_9d.npy`의 translation/rotation convention 및 단위가 Edge AV decoder의 action definition과 일치하는지.
3. pretrained action normalization/denormalization은 이 어댑터에 구현되지 않음. 반드시 학습 전 모델 출력을 같은 GT convention으로 복원할 수 있게 검증.
4. 데이터 loader `smoke`는 GPU forward/backward 검증이 아님.
5. 이 recipe는 Nano LIBERO trainer의 action loss/packer를 Edge 아키텍처에 결합하는 **비공식 적응 코드**. 실제 CUDA 버전에 따라 추가 수정 가능.
6. validation은 GT manifest만 작성하며 실제 trainer 내부 val loop는 아직 활성화하지 않음.

## 본 학습
시험 학습이 성공하면 `--outdir ./runs/stage1_joint_full --max-iters 1000 --epoch-clips 2000` 등으로 **별도 run**을 만드세요. 학습을 이어서 하고자 할 때의 checkpoint 경로는 실제 출력 구조를 확인한 뒤 지정하세요.

## Stage 2
Stage 1 DCP 체크포인트가 만들어진 후:
```bash
python -m cosmos3av.cli plan --stage posttrain --datasets av2 \
  --processed-root "$PROCESSED" \
  --init-checkpoint /실제/stage1/DCP/경로 --wan-vae "$VAE" \
  --outdir ./runs/stage2_av2 --epoch-clips 2000 --max-iters 200 --gpus 4
```
AV2 val 데이터는 학습에 포함하지 않습니다. `--execute`는 다시 smoke/test 후 사용하세요.

## 테스트 (모델 GPU 없이 데이터 선택/설정 생성만)
```bash
pip install -r requirements-planning.txt
PYTHONPATH=. pytest -q tests/test_pipeline.py
```

공식 소스: https://github.com/NVIDIA/cosmos-framework — `edge_model_config.py`, `action_policy_libero_nano.py`, `ActionTransformPipeline`
