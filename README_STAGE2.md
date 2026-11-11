# Cosmos3 Stage2 (AV2) — TOML + W&B patch

This is the **corrected Stage2-only patch**. It does not retrain Stage1. Stage1 is fixed to `runs_stage1_full/s1_raw_v2` (3,000 iterations). It uses the existing Stage1 result at `runs_stage1_full/s1_raw_v2/.../iter_000003000` as initialization.

**Prerequisite:** existing `cosmos3av` / `cosmos3eval` from the previous W&B patch, and the corrected `camera_render.py` already installed in the user's project. This patch adds a new Stage2 shell, a strict TOML parser and a config; it does not overwrite training, evaluation, or visualization Python files.

## Install

From the `cosmos3_ji` project root, unzip `cosmos3_stage2_s1_raw_v2.zip` (the ZIP contains only these new paths):

- `stage2_full_train_eval.sh`
- `scripts/stage2_toml_config.py`
- `configs/stage2_full.toml`
- `README_STAGE2.md`
- `tests/test_stage2_patch.py`

First run `configs/stage2_smoke.toml` for a short 10-iteration initialization test. Once that completes, use `configs/stage2_full.toml` for the full 3,000-iteration run. Edit the TOML before planning. It has **one** `[paths]` table and includes the existing Stage1 paths.

## Commands

```bash
cd /mnt/ufs/users/jaeha-lee/share/jiiyoung/cosmos3_ji
conda activate cosmos
export CONFIG=configs/stage2_smoke.toml

# Check the loaded Stage1 weights without training
bash stage2_full_train_eval.sh --toml "$CONFIG" config
bash stage2_full_train_eval.sh --toml "$CONFIG" inspect

# Stage2 only: AV2
bash stage2_full_train_eval.sh --toml "$CONFIG" plan
bash stage2_full_train_eval.sh --toml "$CONFIG" train

# Post-training checkpoint -> inference weights -> AV2 metrics + rendering
bash stage2_full_train_eval.sh --toml "$CONFIG" export
bash stage2_full_train_eval.sh --toml "$CONFIG" evaluate
```

## W&B

W&B is **disabled by default** in the provided TOML. To enable it, update:

```toml
[wandb]
enabled = true
mode = "online" # or "offline"
```

The training loss is logged through `cosmos3av.cli plan` / Cosmos Framework; `cosmos3eval.cli score` logs `eval/overall/` and `eval/av2/` metrics after inference. Each evaluation uses a separate W&B run. The supported training project is `cosmos3_av_id`. For online, `pip install wandb` and `wandb login` must be completed beforehand.

**Don't change `wandb.mode` between `plan` and `train` for a run**; the script detects changed training parameters.

## Stage1 lineage verification

After `plan`, verify the saved initialization checkpoint:

```bash
python - <<'PY'
import json
r=json.load(open('runs_stage2/s2_av2_baseline_v1/run.json'))
print('stage:',r['stage'])
print('initialized from:',r['init_checkpoint'])
print('Stage1 run:',r['stage1_run'])
print('W&B:',r['wandb_mode'])
PY
```

Before full training: change `CONFIG=configs/stage2_full.toml` and repeat plan/train/export/evaluate.

The script validates `stage1_run/run.json` has `stage=pretrain`, `action_frame=camera` and raw Action, and that the checkpoint has `.metadata` under `model/` and `trainer/`. It saves a SHA256 of Stage1 model checkpoint metadata and verifies it before training. Existing Stage1 outputs are never overwritten.

## Output locations

For the provided config, after successful execution:

- Stage2 DCP: `runs_stage2/s2_av2_baseline_v1/outputs/cosmos3_av_id/posttrain/s2_av2_baseline_v1/checkpoints/iter_000003000/`
- Export: `exports/s2_av2_baseline_v1_iter_3000/`
- AV2 validation selection: `evaluation/stage2_s2_av2_baseline_v1/av2_val_seed42_v1/`
- Prediction: `evaluation/pred_s2_av2_baseline_v1_iter_3000_val_seed42_v1/`
- Metrics: `evaluation/pred_s2_av2_baseline_v1_iter_3000_val_seed42_v1/metrics/summary.json`

## Limitations

This version is intentionally tailored to the existing **raw camera-relative 9D** Stage1 experiment. A custom-normalized Stage1 run is refused until normalization and inverse-normalization are explicitly verified. Checks and mock tests are CPU-only; no GPU or live W&B server execution was performed in this environment.

## Important

- This ZIP does **not** overwrite `configs/stage1_full.toml`, the existing Stage1 weights, or any `cosmos3av/`/`cosmos3eval/` source file.
- If an earlier failed run already created `runs_stage2/<job.run_tag>`, choose a new `run_tag` (e.g., `s2_av2_baseline_v2`), never delete a checkpoint blindly.
- `plan` checks that the Stage1 DCP exists, that it belongs to the configured Stage1 run and that raw camera-relative action contracts match. Actual weight loading must additionally be verified in the GPU training logs.
