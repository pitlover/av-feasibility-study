# Stage1 Reptile 메타 사전학습 (KITTI + Waymo) → Stage2 (AV2)

## 1. 무엇이 달라지나

| | Stage1 TOML | 한 meta-iteration에서 하는 일 | 비고 |
|---|---|---|---|
| 기준선 (joint) | `configs/stage1_full.toml` | KITTI+Waymo를 섞은 배치로 1 step (총 3000 step) | 기존 그대로 (ablation) |
| **Reptile, 두 데이터셋 모두** (`episode_mode = "all_datasets"`) | `configs/stage1_reptile_v1.toml` (v2) | **KITTI와 Waymo 둘 다** 샘플링: 각 데이터셋에서 theta로부터 inner loop k step → 두 변위의 평균으로 theta 갱신 | batched Reptile |
| **Reptile, 한 데이터셋씩** (`episode_mode = "per_dataset"`) | `configs/stage1_reptile_v3.toml` 외 (v3~v10) | **한 데이터셋만** 샘플링 (`alternate`: kitti, waymo 교대 / `random`: `data.weights` 확률): inner loop k step → theta 갱신 | serial Reptile. `support_clips > 0`이면 태스크당 고정 클립을 반복 (cosmos_hs11 방식) |

```
meta-iteration j:
    for task in tasks(j):                      # all_datasets: [kitti, waymo]   per_dataset: [kitti] 또는 [waymo]
        model <- theta
        inner loop: Stage1 optimizer 그대로 (FusedAdam, lr 1e-5, head x5, clip 1.0) k step,
                    매 step 클립 4개 (GPU당 1개)  -> theta_task
    theta <- theta + eps(j) * weighted_mean_task(theta_task - theta)      # eps는 선형 감소
```

- 태스크 = 데이터셋(센서 rig·지역·속도 분포), 샘플 = 61프레임 클립. Stage2(AV2)는 "새 데이터셋에 적응"이므로 inner loop를 Stage2와 같은 optimizer로 돌립니다.
- 쿼리(query) 클립: 서포트와 다른 drive(sequence)에서 뽑은 held-out 클립. 학습에는 쓰지 않고 `query_loss_zero_shot`(theta) vs `query_loss`(적응 후) 진단에만 사용.
- 학습 예산은 cosmos_hs11 로봇 레시피(post-training 1000–2000 iter에 Reptile 1000 meta-iter × k=10)를 따라 기본 **1000 meta-iteration**입니다: all_datasets 1000 × 2 × 10 = 20,000 inner step(기준선 3000 step의 ≈6.7배), per_dataset 1000 × 10 = 10,000(≈3.3배). theta는 meta-iteration마다 `eps × 변위`만큼만 움직이므로(Σeps ≈ 550 / 275 inner loop 상당) 기준선과 step 수를 맞추면 과소학습이 됩니다. inner step 1회 ≈ 기준선 1 step 비용이라 소요 시간 ≈ `max_iters × iter_time`(smoke 로그). query 진단은 10 meta-iter마다 태스크당 forward 2회 추가(≈2%).
- 이 예산에서 KITTI(336 clip)는 per_dataset 약 60 epoch, all_datasets 약 120 epoch 반복됩니다(Waymo 2297 clip은 9/17 epoch). `query_loss_zero_shot/kitti`가 올라가기 시작하면 과적합 신호이므로, 그때는 `dataset_schedule = "random"` + `data.weights` kitti:1, waymo:2~3으로 KITTI 방문 빈도를 낮추세요(`alternate`는 1:1이어야 함).
- eps: all_datasets 1.0→0.1, per_dataset 0.5→0.05(또는 0.25→0.025). 2개 데이터셋에서 "all_datasets 1 step(평균)" ≈ "per_dataset 2 step(kitti 후 waymo)"이므로 compute당 이동량이 같도록 all_datasets의 eps를 2배로 두었습니다.
- 세 번째 모드 `episode_mode = "mixed"`(한 inner loop 안에서 두 데이터셋 배치를 섞음 = joint 샘플링 + Reptile 외부 루프)도 쓸 수 있습니다. 태스크 구조 없이 외부 루프만의 효과를 보는 컨트롤입니다.
- 비교 대상: (a) Stage 1 없이 Stage 2만(Edge 기본 레시피), (b) joint Stage 1(`stage1_full.toml`, 동업자 표준 3000 step), (c) Reptile. (b)는 Reptile보다 step 수가 적은 "레시피 기준선"이고, step-matched joint 컨트롤이 필요하면 `stage1_full.toml`을 복사해 `train.max_iters = 10000`으로 돌리거나, hs11의 k=1/eps=1 컨트롤처럼 `[reptile] episode_mode="mixed", inner_steps=1, inner_warmup_steps=0, inner_reset_optimizer=false, meta_lr=1.0, meta_lr_min_ratio=1.0, max_iters=10000`으로 두면 per_dataset 변형과 같은 클립 스트림에서 순수 joint 학습이 됩니다.

### 현재 스윕 (`configs/stage1_reptile_v*.toml`, 이 커밋 시점; 각 파일의 헤더와 `[reptile]` 주석이 기준)

| TOML | run_tag | episode_mode / schedule | k | eps | support | max_iters |
|---|---|---|---|---|---|---|
| v1 | `s1_reptile_v1` | all_datasets | 10 | 1.0→0.1 | 매 step 새 클립 | 1000 |
| v2 | `s1_reptile_v2` | all_datasets | 10 | 1.0→0.1 | 새 클립 | 500 |
| v3 | `s1_reptile_v3` | per_dataset / alternate | 10 | 0.5→0.05 | 새 클립 | 1000 |
| v4 | `s1_reptile_v4` | per_dataset / alternate | 10 | 0.5→0.05 | 새 클립 | 500 |
| v5 | `s1_reptile_v5` | per_dataset / alternate | 10 | 0.25→0.025 | 새 클립 | 1000 |
| v6 | `s1_reptile_v6` | per_dataset / alternate | 20 | 0.25→0.025 | 새 클립 | 1000 |
| v7 | `s1_reptile_v7` | per_dataset / alternate | 10 | 0.25→0.025 | 새 클립 | 500 |
| v8 | `s1_reptile_v8` | per_dataset / alternate | 10 | 0.5→0.05 | 새 클립 | 2000 |
| v9 | `s1_reptile_v9` | per_dataset / random | 10 | 0.5→0.05 | 16 클립 고정 (hs11 방식) | 1000 |
| v10 | `s1_reptile_v10` | per_dataset / random | 10 | 0.25→0.025 | 16 클립 고정 | 1000 |

TOML을 복사해 변형을 만들 때 `job.run_tag`는 파일마다 달라야 하고(쉘이 기존 run dir를 거부), `train.max_iters`는 `save_every`의 배수여야 합니다. `tests/test_stage1_reptile_toml.py`가 `configs/stage1_reptile_*.toml` 전체의 파싱·run_tag 중복·구조 조건을 검사합니다.

### 파일

| 파일 | 변경 |
|---|---|
| `configs/stage1_reptile_v1.toml` ~ `v10.toml`, `stage1_reptile_smoke.toml` | 신규 |
| `cosmos3av/reptile.py` (설정·에피소드 샘플러), `reptile_meta.py` (theta 보관·meta step), `reptile_data.py` (클립 스트림), `reptile_train.py` (학습 루프), `entry_reptile.py` (torchrun 진입점) | 신규 |
| `cosmos3av/cli.py` | `plan --reptile-config` (run.json `trainer=reptile`, `reptile_config.json`, `train_unique.jsonl`, sft_config.toml에 `[model.ema] enabled=false`), `launch`가 trainer에 따라 진입점 선택 |
| `scripts/stage1_toml_config.py` | `[reptile]` 키 + 검증, `--format reptile-json` |
| `stage1_full_train_eval.sh` | `plan`이 `[reptile]`을 planner에 전달, plan/train 설정 고정 검사에 `reptile.*` 포함. 경로 구조는 그대로 |
| `tests/test_reptile_*.py`, `tests/test_stage1_reptile_*.py` | CPU 테스트 |

`cosmos3eval/`, `stage2_full_train_eval.sh`, `configs/stage2_full.toml`은 건드리지 않았습니다.

## 2. Stage1 실행 (기존과 같은 명령, TOML만 다름)

먼저 smoke (4 meta-iter × 2 step, W&B off, 새 run_tag):

```bash
export S1_CONFIG=configs/stage1_reptile_smoke.toml
bash stage1_full_train_eval.sh --toml "$S1_CONFIG" config     # trainer=reptile, [reptile] 값 출력
bash stage1_full_train_eval.sh --toml "$S1_CONFIG" plan
bash stage1_full_train_eval.sh --toml "$S1_CONFIG" train      # DCP: .../checkpoints/iter_000000004/
bash stage1_full_train_eval.sh --toml "$S1_CONFIG" export
bash stage1_full_train_eval.sh --toml "$S1_CONFIG" evaluate
```

본 실험:

```bash
export S1_CONFIG=configs/stage1_reptile_v1.toml    # 또는 다른 stage1_reptile_v*.toml
bash stage1_full_train_eval.sh --toml "$S1_CONFIG" plan
bash stage1_full_train_eval.sh --toml "$S1_CONFIG" train
bash stage1_full_train_eval.sh --toml "$S1_CONFIG" export
bash stage1_full_train_eval.sh --toml "$S1_CONFIG" evaluate
```

인자 없는 실행 (`configs/scripts/run_stage1_reptile_<smoke|v1..v10>.sh`): 각 파일은 자기 TOML 하나만 가리키는 얇은 wrapper이고, 단계는 TOML의 `[job] run_step`(`config|inspect|plan|train|export|prepare|oracle|infer|score|render|evaluate|all`, 생략 시 `config` = 설정 출력만)에서 읽어 `stage1_full_train_eval.sh --toml <그 TOML>`에 그대로 위임합니다. 프로젝트 루트에서 `bash configs/scripts/run_stage1_reptile_v1.sh`처럼 실행하면 되고(파일을 루트로 복사해도 동작), 검사·출력 구조·Reptile 라우팅은 위 명령과 완전히 같습니다. smoke TOML은 `run_step = "all"`이라 한 번에 plan→train→export→evaluate를 돕니다. 출력 루트를 옮기고 싶으면 `[paths] run_root / exports_root / evaluation_root`(기본 `runs_stage1_full`, `exports`, `evaluation`; 상대 경로는 `paths.base` 아래)를 지정합니다.

출력 경로는 기존 규칙 그대로이고, `<tag>` = 그 TOML의 `job.run_tag`, `<N>` = `train.max_iters`(meta-iteration 수; 기준선의 3000과 다릅니다). v1이면 `<tag>` = `s1_reptile_v1`, `<N>` = 1000:

| | 경로 |
|---|---|
| run dir | `runs_stage1_full/<tag>/` |
| 최종 DCP | `runs_stage1_full/<tag>/outputs/cosmos3_av_id/pretrain/<tag>/checkpoints/iter_<N을 9자리로>/` (v1: `iter_000001000/`) |
| export | `exports/<tag>_iter_<N>/` (v1: `exports/s1_reptile_v1_iter_1000/`) |
| 정량 | `evaluation/pred_<tag>_iter_<N>_val_seed42_v1/metrics/summary.json` |
| 정성 | `evaluation/pred_<tag>_iter_<N>_val_seed42_v1/qualitative/{kitti,waymo}/` |

`save_every`는 기본적으로 `max_iters`(최종 DCP만)입니다. 한 run이 여러 시간이면 250/500으로 낮추세요. auto-resume은 저장된 DCP에서만 이어가며, DCP 하나는 기준선 DCP와 비슷한 크기입니다.

- W&B는 offline (`[wandb] enabled=true, mode="offline"`): `runs_stage1_full/<tag>/wandb/` 아래 `offline-run-*` → `wandb sync <폴더>`. 같은 수치가 `.../pretrain/<tag>/reptile_train_log.jsonl`에도 매 meta-iteration 기록됩니다.
- 주요 로그 키: `support_loss_first → support_loss_last` (inner loop 손실), `query_loss_zero_shot` vs `query_loss` (`query_loss_gain` = 차이), `adapt_delta/<group>` (inner loop가 theta를 얼마나 움직였나), `eps`, `meta_step_norm`, `theta_norm/<group>`, `mem_peak_alloc_gb`.
- 재시작: 같은 `train` 명령을 다시 실행하면 `latest_checkpoint.txt`에서 auto-resume 됩니다 (쉘이 `train.log`가 있으면 거부하므로 먼저 옮기거나 지울 것).
- 설정을 바꾸려면 TOML의 `[reptile]`만 수정하면 됩니다. 모든 키 설명은 `cosmos3av/reptile.py` 상단 docstring과 TOML 주석에 있습니다. `[reptile]` 값은 plan 때 고정되어 `train` 시 바뀌면 `[CONFIG MISMATCH]`로 거부됩니다.

## 3. Stage2 시작 방법 — 달라지는 것은 TOML 세 줄

Stage2 코드·레시피는 기존 그대로입니다. Stage2 스크립트가 Stage1 run에서 읽는 것은 `run.json`(stage=pretrain, action_frame=camera, action_normalizer=null, wan_vae)과 DCP의 `model/.metadata`·`trainer/.metadata`, 그리고 체크포인트가 `stage1_run/outputs` 아래에 있는지뿐인데, Reptile run dir은 이를 그대로 만족합니다. `configs/stage2_full.toml`을 복사해서 세 줄만 바꿉니다.

```bash
cp configs/stage2_full.toml configs/stage2_reptile_v1.toml
sed -i 's/^run_tag = "s2_av2_pipeline_v1"/run_tag = "s2_av2_reptile_v1"/; s#s1_pipeline_v1#s1_reptile_v1#g; s/iter_000003000/iter_000001000/' configs/stage2_reptile_v1.toml
grep -n 'run_tag\|stage1_' configs/stage2_reptile_v1.toml
```

결과(`configs/stage2_reptile_v1.toml`):

```toml
[job]
run_tag = "s2_av2_reptile_v1"          # 새 Stage2 run 이름 (기존 s2_av2_pipeline_v1과 구분)

[paths]
stage1_run = "runs_stage1_full/s1_reptile_v1"
stage1_checkpoint = "runs_stage1_full/s1_reptile_v1/outputs/cosmos3_av_id/pretrain/s1_reptile_v1/checkpoints/iter_000001000"
```

다른 변형은 `s1_reptile_v1` 대신 그 TOML의 `job.run_tag`, `iter_000001000` 대신 그 TOML의 `train.max_iters`(v2/v4/v7: `iter_000000500`, v8: `iter_000002000`, smoke: `iter_000000004`)를 넣습니다. Stage1 run마다 Stage2 TOML을 하나씩 만들고 `job.run_tag`도 각각 다르게 두세요. 나머지는 그대로 두고 기존 명령을 실행합니다.

```bash
export S2_CONFIG=configs/stage2_reptile_v1.toml
bash stage2_full_train_eval.sh --toml "$S2_CONFIG" config      # "[CONFIG] resolved Stage1 DCP: .../s1_reptile_v1/.../iter_000001000"
bash stage2_full_train_eval.sh --toml "$S2_CONFIG" inspect
bash stage2_full_train_eval.sh --toml "$S2_CONFIG" plan        # "[STAGE1 CHECK] source checkpoint: ..." / "[PLAN OK] Stage2 initial checkpoint is Stage1: ..."
bash stage2_full_train_eval.sh --toml "$S2_CONFIG" train
bash stage2_full_train_eval.sh --toml "$S2_CONFIG" export
bash stage2_full_train_eval.sh --toml "$S2_CONFIG" evaluate
```

Stage2 출력: DCP `runs_stage2/s2_av2_reptile_v1/outputs/cosmos3_av_id/posttrain/s2_av2_reptile_v1/checkpoints/iter_000003000/`, export `exports/s2_av2_reptile_v1_iter_3000/`, 정량 `evaluation/pred_s2_av2_reptile_v1_iter_3000_val_seed42_v1/metrics/summary.json`, 정성 `.../qualitative/av2/`.

- Stage2 `plan`은 `paths.stage1_run`, `paths.stage1_checkpoint`, `job.run_tag`를 `resolved_stage2_config.json`에 고정하고 Stage1 DCP `model/.metadata`의 sha256을 저장하므로, 이후 `train`은 같은 TOML로 실행해야 합니다.
- Reptile DCP에는 `net_ema.*`가 없습니다(EMA 끔). Stage2는 `keys_to_skip_loading=["net_ema."]`로 `net.*`만 로드하므로 문제 없고, joint Stage1 DCP에서 시작할 때도 Stage2가 읽는 것은 raw `net.*`뿐이라 초기화 경로는 동일합니다. Stage2 학습 자체는 기존대로 EMA on, lr 1e-5, 3000 iter입니다.
- Stage2의 W&B는 Stage2 TOML의 `[wandb]`(기본 disabled)를 따릅니다. offline으로 남기려면 복사한 TOML에서 `enabled = true`, `mode = "offline"`.
- Stage1 없이 Stage2만 돌리는 기준선(Cosmos3-Edge 기본 레시피)은 이 스크립트가 `stage1_run/run.json`을 요구하므로 그대로는 안 되고, 기존처럼 `cosmos3av.cli plan --stage posttrain --init-checkpoint <Edge DCP>`(`--stage1-run` 없이)로 따로 돌려야 합니다.

## 4. `[reptile]` 키

| 키 | v1 / v3 | 의미 |
|---|---|---|
| `enabled` | true | false면 기존 joint 학습 (stage1_full.toml과 동일 동작) |
| `episode_mode` | `all_datasets` / `per_dataset` | 한 meta-iteration의 태스크: 모든 데이터셋 / 하나 / `mixed`(섞은 배치 하나) |
| `dataset_schedule` | – / `alternate` | per_dataset에서 데이터셋 선택: 교대 또는 `random`(data.weights 확률; v9·v10) |
| `inner_steps`, `inner_warmup_steps` | 10, 2 | 태스크당 optimizer step 수, 시작 LR 램프 길이 |
| `inner_grad_clip`, `inner_reset_optimizer` | 1.0, true | 기준선과 같은 clip; 태스크마다 Adam 상태 초기화 |
| `meta_lr`, `meta_lr_min_ratio`, `meta_lr_warmup_iters` | 1.0/0.1 · 0.5/0.1 · 0 | eps 시작값, 끝값 비율(`max_iters`에서 도달), 워밍업 |
| `clips_per_rank` | 1 | GPU당 step당 클립 수 (기준선의 `max_samples_per_batch`와 동일) |
| `support_clips` | 0 | 0 = 매 step 새 클립. 양수 = 태스크당 고정 클립 수를 반복(few-shot 식, hs11 방식; v9·v10은 16) |
| `query_clips_per_rank`, `query_sequence_disjoint`, `eval_query_every` | 1, true, 10 | 진단용 held-out 클립 |
| `loss_mode` | `total` | 기준선과 같은 목적함수 (`action` = action 항만) |

`data.weights`: all_datasets에서는 두 변위의 평균 가중치, `random` 스케줄에서는 데이터셋 선택 확률, `mixed`에서는 배치 내 비율 (`alternate`는 1:1이어야 함).

## 5. 주의

- GPU 검증은 하지 않았습니다 (CPU 테스트만: 샘플러·meta step 수식·plan/launch·실제 Cosmos 로더 스택으로 배치 1 에피소드). 반드시 `stage1_reptile_smoke.toml`부터 돌리세요.
- 메모리: EMA 사본 대신 theta 사본(학습 파라미터의 fp32 shard, GPU당 ≈1.4 GB)을 둡니다. 기준선이 돌던 GPU면 같은 자리에서 돌 것으로 예상하지만, smoke 로그의 `peak mem`으로 확인하세요. `entry_reptile.py`가 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`를 기본으로 켭니다(환경변수로 덮어쓰기 가능).
- `train.max_iters`를 줄이면 eps 스케줄도 거기서 끝나도록 잡히므로, 중간에 끊지 말고 `max_iters` 자체를 바꾸세요. `save_every`는 meta-iteration 단위이며 DCP 하나는 기준선과 비슷한 크기입니다.
- CPU 테스트: `python -m unittest discover -s tests -p 'test_reptile_*.py'` 및 `-p 'test_stage1_reptile_*.py'` (cosmos 환경에서; 프레임워크가 없으면 로더 테스트는 자동 skip).
