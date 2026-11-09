# Cosmos3 Stage1 → Stage2 TOML 통합 실행 가이드

## 목적

- Stage1: `Cosmos3-Edge Base → KITTI+Waymo` (camera-relative raw 9D, 61 RGB frames → 60 actions)
- Stage2: `Stage1 DCP → Argoverse2 posttrain` (동일한 Action 정의)
- 단계마다 TOML 하나로 `plan`, `train`, `export`, `evaluate`를 실행
- 평가 유형은 `[eval].mode = "full" | "score" | "render" | "both"`
- W&B `[wandb].enabled = false`가 기본값. `true`면 `mode="online"` 또는 `"offline"` 지정

## 파일과 적용

패키지 내 파일:

```
stage1_full_train_eval.sh
stage2_full_train_eval.sh
scripts/stage1_toml_config.py
scripts/stage2_toml_config.py
scripts/choose_render_clips.py
configs/stage1_full.toml
configs/stage2_full.toml
tests/test_toml_modes.py
```

서버 프로젝트 루트에서 기존 파일을 백업하고 압축을 해제하세요. **기존 TOML 파일을 손수 수정했다면 백업하고 비교한 뒤 병합**하세요. 이 패키지는 `cosmos3av/`, `cosmos3eval/`, `camera_render.py`를 덮어쓰지 않습니다. 기존 W&B 평가 통합 패치와 61프레임 시각화 수정이 적용되어 있어야 합니다.

```bash
cd /mnt/ufs/users/jaeha-lee/share/jiiyoung/cosmos3_ji
cp stage1_full_train_eval.sh stage1_full_train_eval.sh.before_unified
cp stage2_full_train_eval.sh stage2_full_train_eval.sh.before_unified
cp configs/stage1_full.toml configs/stage1_full.toml.before_unified
cp configs/stage2_full.toml configs/stage2_full.toml.before_unified
unzip -o cosmos3_stage1_stage2_unified.zip
bash -n stage1_full_train_eval.sh && bash -n stage2_full_train_eval.sh
```

## Stage1: 학습 → Export → 평가

TOML: `configs/stage1_full.toml`.

```bash
export S1_CONFIG=configs/stage1_full.toml
bash stage1_full_train_eval.sh --toml "$S1_CONFIG" config
bash stage1_full_train_eval.sh --toml "$S1_CONFIG" inspect
bash stage1_full_train_eval.sh --toml "$S1_CONFIG" plan
bash stage1_full_train_eval.sh --toml "$S1_CONFIG" train
bash stage1_full_train_eval.sh --toml "$S1_CONFIG" export
bash stage1_full_train_eval.sh --toml "$S1_CONFIG" evaluate
```

기본 TOML의 `[eval].mode="full"`에서는 KITTI/Waymo validation 선별·GT Oracle·추론·정량평가·정성평가가 순서대로 실행됩니다. 기본적으로 KITTI 3개 + Waymo 3개 영상이 그려집니다. `render_enabled=false`면 **full 모드에서** Render만 생략됩니다.

Stage1 최종 DCP:

```
runs_stage1_full/s1_pipeline_v1/outputs/cosmos3_av_id/pretrain/s1_pipeline_v1/checkpoints/iter_000003000/
```

Stage1 추론용 Export:

```
exports/s1_pipeline_v1_iter_3000/
```

Stage1 평가 데이터/결과:

```
evaluation/stage1_s1_pipeline_v1/kitti_waymo_val_seed42_v1/
evaluation/pred_s1_pipeline_v1_iter_3000_val_seed42_v1/metrics/summary.json
evaluation/pred_s1_pipeline_v1_iter_3000_val_seed42_v1/qualitative/kitti/
evaluation/pred_s1_pipeline_v1_iter_3000_val_seed42_v1/qualitative/waymo/
```

## Stage2: 동일한 Stage1 DCP에서 시작

TOML: `configs/stage2_full.toml`.

```
paths.stage1_run = "runs_stage1_full/s1_pipeline_v1"
paths.stage1_checkpoint = "runs_stage1_full/s1_pipeline_v1/outputs/cosmos3_av_id/pretrain/s1_pipeline_v1/checkpoints/iter_000003000"
```

Stage1에 다른 `run_tag`를 사용했다면 **위 두 경로를 같이 변경**해야 합니다. Stage2 `plan`은 `run.json`의 `stage=pretrain`, `action_frame=camera`, `action_normalizer=null` 및 DCP 메타데이터 존재 여부를 확인합니다. Stage1 체크포인트만 존재해도 `run.json`이 없으면 Stage2 `plan`을 진행하지 않습니다. 이는 잘못된 체크포인트를 조용히 사용하는 일을 방지하기 위한 안전장치입니다.

```bash
export S2_CONFIG=configs/stage2_full.toml
bash stage2_full_train_eval.sh --toml "$S2_CONFIG" config
bash stage2_full_train_eval.sh --toml "$S2_CONFIG" inspect
bash stage2_full_train_eval.sh --toml "$S2_CONFIG" plan
bash stage2_full_train_eval.sh --toml "$S2_CONFIG" train
bash stage2_full_train_eval.sh --toml "$S2_CONFIG" export
bash stage2_full_train_eval.sh --toml "$S2_CONFIG" evaluate
```

Stage2 최종 DCP:

```
runs_stage2/s2_av2_pipeline_v1/outputs/cosmos3_av_id/posttrain/s2_av2_pipeline_v1/checkpoints/iter_000003000/
```

Stage2 평가:

```
evaluation/stage2_s2_av2_pipeline_v1/av2_val_seed42_v1/
evaluation/pred_s2_av2_pipeline_v1_iter_3000_val_seed42_v1/metrics/summary.json
evaluation/pred_s2_av2_pipeline_v1_iter_3000_val_seed42_v1/qualitative/av2/
```

## TOML 평가 모드 선택 (Stage1, Stage2 동일)

**처음 평가할 때** `[eval].mode="full"`을 선택합니다. 추론용 Export가 생성된 상태여야 합니다. 이미 동일한 Pred 폴더에 결과가 있으면 새 추론을 거부하므로, 기존 결과를 이용할 때는 아래 모드를 선택하세요.

| eval.mode | 실행 동작 | 새로운 추론? |
|---|---|---|
| `full` | Prepare → Oracle → Infer → Score → optional Render | 예 |
| `score` | 기존 Pred를 읽어서 RRE/RTE/ATE 계산 | 아니요 |
| `render` | 기존 Pred의 GT/Pred Overlay + BEV 생성 | 아니요 |
| `both` | 기존 Pred를 읽어서 Score → Render | 아니요 |

TOML 파일을 편집하거나 CLI에서 일시적으로 덮어쓸 수 있습니다.

```bash
# 기존 Stage1 예측 결과로 정량 평가만
bash stage1_full_train_eval.sh --toml "$S1_CONFIG" evaluate eval.mode=score
# 기존 Stage1 예측 결과로 정성 평가만
bash stage1_full_train_eval.sh --toml "$S1_CONFIG" evaluate eval.mode=render
# 기존 Stage2 예측 결과로 정량+정성 평가
bash stage2_full_train_eval.sh --toml "$S2_CONFIG" evaluate eval.mode=both
```

별도 `*_retry1` 폴더의 예측을 보려면 TOML `[eval]`의 `eval_dir`, `pred_dir`를 지정합니다. 그 폴더가 실제 동일한 평가 클립 `selection.jsonl`을 사용해야 합니다. `score/render/both`는 선택된 **모든** 클립의 Pred가 있는지 먼저 확인합니다. 일부 누락되면 조용히 건너뛰지 않고 오류를 냅니다.

```
[eval]
mode = "both"
eval_dir = "evaluation/stage2_s2_smoke_debug_01/av2_smoke_debug_val_01"
pred_dir = "evaluation/pred_s2_smoke_debug_01_retry1"
render_enabled = true
render_max_clips = 3
render_clip_ids = []
ground_height_m = 1.7
horizon_steps = 60
fps = 10
layout = "both"
```

`render_clip_ids`로 특정 Scene을 지정할 수도 있습니다. 해당 Scene은 평가 목록에 포함되어야 합니다. `render_max_clips`는 Stage1에서 **데이터셋별** 개수, Stage2에서는 AV2 개수입니다. `ground_height_m=1.7`은 영상 투영용 지면 가정이며 RRE/RTE/ATE 계산에는 적용되지 않습니다.

## W&B

```
[wandb]
enabled = false
mode = "disabled"
project = "cosmos3_av_id"
```

Online:

```
[wandb]
enabled = true
mode = "online"
project = "cosmos3_av_id"
```

Offline은 `mode="offline"`. `online`에서는 서버에 `pip install wandb; wandb login`이 필요합니다. **학습 Loss는 학습 중** Cosmos Framework에서, **RRE/RTE/ATE는 `score`가 실행될 때** 별도 W&B 평가 Run에 기록합니다. 학습 W&B Project 이름은 현재 Cosmos Framework 어댑터에서 `cosmos3_av_id`로 고정되어 있으며 `wandb.project`는 평가 Run에 적용됩니다.

## 짧은 Smoke Test

실제 GPU 환경에서 먼저 새 Run으로 10 iteration만 확인하려면 Stage1과 Stage2에 각각 다른 `job.run_tag`를 설정하고 두 TOML의 Stage1 경로도 맞추세요. 두 TOML 모두 `train.max_iters=10`, `train.save_every=10`으로 변경하고 Stage2의 `paths.stage1_checkpoint` 끝을 `iter_000000010`으로 바꿉니다. 본 학습 결과와 동일한 `run_tag`를 사용하지 마세요.

## 검증 범위

Python TOML 파싱, Bash 문법 검사, 스텁 기반 `plan → train → export → full` 및 `score/render/both` 분기 테스트를 수행했습니다. **실제 서버 GPU 및 Cosmos Framework 실행은 이 패키지를 사용하여 검증하지 않았습니다.** 61프레임 `camera_render.py` 수정본과 이전 W&B 평가 연동이 유지되어야 합니다.
