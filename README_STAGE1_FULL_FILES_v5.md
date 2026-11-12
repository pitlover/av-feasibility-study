# Stage-1: KITTI / Waymo / Argoverse2 / nuScenes 지원 (전체 수정 파일 포함)

이 ZIP은 **기존 저장소 전체 복제본은 아니며**, `feat/prepare_nuscenes` 브랜치에서 **수정해야 할 원본 소스 전체(8개)와 신규 유틸리티, TOML, 테스트 코드**를 담은 덮어쓰기용 ZIP입니다. `apply_stage1_multisource.py` 실행은 **필요하지 않습니다**.

**중요:** 사용자가 공유한 Cosmos Framework와 `/mnt/ufs` 데이터에 대한 실제 GPU 학습 검증은 수행하지 못했습니다. 로컬 정적 코드 검사, 쉘 문법 검사, synthetic 4 데이터셋 × 10/12/20Hz 인덱스 테스트는 통과했습니다. `plan`에서 공식 `ActionTransformPipeline` smoke test를 통과하고 소규모 GPU 학습을 검증한 후 전체 학습을 진행하세요. K는 전처리 메타데이터에 존재하지만 현재 Stage-1 모델 입력 텐서로 직접 전달하지 않습니다.

## 무엇이 들어 있나

기존 모듈(통째로 교체):
- `cosmos3av/data.py`: nuScenes/AV2를 Stage-1에 허용, 10/12/20Hz 클립 검증
- `cosmos3av/framework_dataset.py`: 클립별 fps를 conditioning_fps로 전달
- `cosmos3av/cli.py`: Stage-1 계획/통계 데이터셋 선택 확장
- `cosmos3av/experiment.py`: 정책 메타데이터의 FPS 계약 설명 및 clip-specific fps 대응
- `cosmos3eval/data.py`: 10/12/20Hz 검증, inference request fps 반영
- `cosmos3eval/cli.py`: 평가 CLI에서 nuScenes 지원
- `scripts/stage1_toml_config.py`: TOML `data.datasets`, `data.weights`, `data.sources` 지원
- `stage1_full_train_eval.sh`: 선택 데이터셋을 inspect/plan/train/evaluate에서 실제 사용

추가 파일:
- `scripts/stage1_source_index.py`: 데이터셋별 root를 해석해 하나의 학습 인덱스 생성. 원본 데이터는 불변
- `configs/stage1_virtual_10hz.toml`: KITTI + Waymo + AV2, 모두 10Hz/가상 카메라
- `configs/stage1_virtual_rawhz.toml`: 네 데이터셋, 가상 카메라, 10/12/20Hz
- `configs/stage1_original_rawhz.toml`: 네 데이터셋, 원본 카메라, 10/12/20Hz
- `configs/stage1_original_10hz.toml`: KITTI + Waymo + AV2, 원본 카메라, 10Hz
- `configs/stage1_mixed_sources.toml`: nuScenes만 가상 12Hz 별도 경로
- `configs/stage1_mixed_original_nuscenes.toml`: nuScenes만 원본 카메라/12Hz 별도 경로
- `tests/test_stage1_multisource_source.py`: synthetic index 테스트

## 설치 (기존 서버 checkout 루트에서)

```bash
cd /mnt/ufs/users/jaeha-lee/share/jiiyoung/av-feasibility-study
# 현재 브랜치와 로컬 변경 확인
git branch --show-current
git status --short

# 필요하면 먼저 별도 백업
cp cosmos3av/data.py cosmos3av/data.py.bak
cp cosmos3av/cli.py cosmos3av/cli.py.bak
cp stage1_full_train_eval.sh stage1_full_train_eval.sh.bak

# zip 파일을 프로젝트 루트에 복사했다고 가정
unzip -o cosmos3_stage1_full_modified_sources_v5.zip -d .
# 주의: 이 ZIP을 풀고 나면 기존 코드 8개가 즉시 교체됩니다.
# 별도의 패치 --check / --apply 명령을 실행하지 마세요.
```

## 확인 및 실행

```bash
python3 -m compileall -q cosmos3av cosmos3eval scripts
bash -n stage1_full_train_eval.sh
python3 -m unittest discover -s tests -p "test_stage1_multisource_source.py" -v
bash stage1_full_train_eval.sh --toml configs/stage1_virtual_10hz.toml config
bash stage1_full_train_eval.sh --toml configs/stage1_virtual_10hz.toml inspect
bash stage1_full_train_eval.sh --toml configs/stage1_virtual_10hz.toml plan
# plan과 같은 TOML 사용
bash stage1_full_train_eval.sh --toml configs/stage1_virtual_10hz.toml train
```

## 핵심 TOML 설정

```toml
[paths]
processed = "/mnt/ufs/users/jaeha-lee/share/jiiyoung/av-feasibility-study/processed_virtual_native_v2"

[data]
datasets = ["kitti", "waymo", "av2"]
epoch_clips = 12000
seed = 42

[data.weights]
kitti = 1.0
waymo = 1.0
av2 = 1.0

[data.sources]
# 개별 데이터셋만 다른 root에서 로드하고 싶을 때 지정
# nuscenes = "/mnt/ufs/users/jaeha-lee/share/jiiyoung/cosmos3_ji/processed_virtual_rawhz"
```

**참고:** `processed_virtual_native_v2`에는 nuScenes가 없고, nuScenes는 `processed_virtual_rawhz/virtual_native` (12Hz) 또는 `processed_raw_rawhz/original_native` (12Hz) 중 하나를 선택해야 합니다. 원본 FPS 설정의 AV2는 20Hz로, KITTI/Waymo는 10Hz로 기록됩니다. 전부 61프레임이지만 10Hz는 약 6초, 12Hz는 약 5초, 20Hz는 약 3초에 해당합니다.

## 검증 범위

`python -m compileall`, `bash -n`, 그리고 4개 데이터셋 및 경로 이동에 대한 synthetic 테스트를 완료했습니다. 실제 `/mnt/ufs` 디렉터리 및 GPU/Cosmos Framework의 학습은 이 환경에서 재현할 수 없어 검증하지 못했습니다. Run root 설정과 데이터셋 split을 `inspect`에서 반드시 확인하세요.

**이 패키지를 만들기 위해 GitHub에 push하지 않았습니다.**
