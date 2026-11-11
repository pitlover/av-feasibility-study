# Cosmos3-Edge AV: optional Weights & Biases integration

## What changes

- **Training:** `cosmos3av.cli plan --wandb-mode disabled|online|offline` puts `[job].wandb_mode` into `sft_config.toml`. Cosmos Framework's **native** W&B callback logs its training loss and iteration/optimizer metrics. No custom second training `wandb.init` is created.
- **Evaluation:** `cosmos3eval.cli score --wandb-mode ...` logs `eval/overall/*`, `eval/kitti/*`, `eval/waymo/*`, and `eval/av2/*` as scalar metrics. Logs a per-clip W&B Table (up to 5,000 clips) and versioned CSV/JSON evaluation artifact. GT Oracle is intentionally not logged as model performance.
- **Association:** evaluation is a **separate W&B run** in the same `project=cosmos3_av_id`, `group=pretrain` or `posttrain`. Training steps and summary evaluation metrics are not mixed into one time series.
- **Default:** `disabled`; no SDK import or W&B network call by this project code. No evaluation history is uploaded unless explicitly enabled.

## Apply without overwriting existing processing or visualization

1. Back up your existing `cosmos3av/cli.py` and `cosmos3eval/cli.py` plus any shell scripts you are replacing.
2. From this patch zip, copy ONLY these files into the matching project paths:
   - `cosmos3av/cli.py`
   - `cosmos3eval/cli.py`
   - `cosmos3eval/wandb_logging.py` (new)
   - `requirements-wandb.txt` (new)
   - optionally `stage1_full_train_eval.sh`, `retrain_cosmos3_av_repro_v2.sh`, `tests/test_wandb_optional.py`
3. **Do not replace `cosmos3eval/camera_render.py`**; this patch does not include it. Existing models, data, predictions, and checkponts remain unchanged.

```bash
cd /mnt/ufs/users/jaeha-lee/share/jiiyoung/cosmos3_ji
conda activate cosmos
pip install -r requirements-wandb.txt
python -m pytest -q tests/test_wandb_optional.py
python -m cosmos3av.cli plan --help | grep wandb
python -m cosmos3eval.cli score --help | grep wandb
```

## Existing Stage1: upload evaluation metrics now without retraining

```bash
WANDB_MODE=offline bash stage1_full_train_eval.sh score
# or:
WANDB_MODE=online bash stage1_full_train_eval.sh score
```

This does NOT recover training loss from the old already-finished run; native W&B loss logging starts when training is launched with W&B enabled.

## New Stage1 training with W&B

Offline (no live upload):

```bash
WANDB_MODE=offline RUN_TAG=s1_raw_wandb_v1 bash stage1_full_train_eval.sh plan
WANDB_MODE=offline RUN_TAG=s1_raw_wandb_v1 bash stage1_full_train_eval.sh train
RUN_TAG=s1_raw_wandb_v1 bash stage1_full_train_eval.sh export
WANDB_MODE=offline RUN_TAG=s1_raw_wandb_v1 bash stage1_full_train_eval.sh evaluate
```

Online (configure credentials with `wandb login`, or export `WANDB_API_KEY` securely):

```bash
WANDB_MODE=online RUN_TAG=s1_raw_wandb_online_v1 bash stage1_full_train_eval.sh plan
WANDB_MODE=online RUN_TAG=s1_raw_wandb_online_v1 bash stage1_full_train_eval.sh train
RUN_TAG=s1_raw_wandb_online_v1 bash stage1_full_train_eval.sh export
WANDB_MODE=online RUN_TAG=s1_raw_wandb_online_v1 bash stage1_full_train_eval.sh evaluate
```

In either case, `plan` persists mode in `run.json` and `sft_config.toml`; `train` follows it even if `WANDB_MODE` is omitted.

### Direct training CLI

Append `--wandb-mode online|offline|disabled` to your existing `cosmos3av.cli plan ...` command. The default is disabled. A `launch` command uses the planned value, or override it without editing the original TOML:

```bash
python -m cosmos3av.cli launch \
  --run-dir runs_stage1_full/s1_new \
  --framework-root cosmos-framework \
  --wandb-mode offline --execute
```

Use a new run directory if training again. Don't relaunch into an existing completed checkpoint directory.

## Explicit evaluation CLI (works with existing predictions)

```bash
python -m cosmos3eval.cli score \
  --eval-dir evaluation/stage1_s1_full_raw_v1/kitti_waymo_val_seed42_v1 \
  --pred-dir evaluation/pred_s1_full_raw_v1_iter_3000_val_seed42_v1 \
  --model s1_full_raw_v1_iter_3000 \
  --wandb-mode offline --wandb-project cosmos3_av_id \
  --wandb-group pretrain --wandb-name eval_s1_full_raw_v1_iter_3000
```

For Stage2 AV2 score, use `--wandb-group posttrain` and the appropriate AV2 evaluation and prediction directories. The optional `retrain_cosmos3_av_repro_v2.sh` wrapper forwards the selected mode to both stages and evaluation.

### Offline sync

Training files: `runs_stage1_full/<RUN_TAG>/wandb/` (if native framework honors `WANDB_DIR`).

Evaluation files: `<PRED_DIR>/metrics/wandb/`.

```bash
find runs_stage1_full evaluation -type d -name 'offline-run-*' -print
wandb sync PATH_TO_OFFLINE_RUN_FOLDER
```

Offline sync sends results later, when connected and authenticated. Do not commit `wandb` directories or API keys into Git.

## Validation / scope

- CPU regression suite (`pytest`) plus `bash -n` passed on this patch; no live GPU training or real W&B API upload was executed in this environment.
- This patch is based on the **uploaded** `cosmos3_edge_av_full_code_training_cli_fixed(1).zip` and companion user scripts; your server may have newer local edits. Review differences before copying those two existing CLI modules.
- Eval metrics are added **only after score successfully writes the metrics files**. If W&B is enabled but SDK/auth/network fails, local metrics still exist; the `score` command returns an error, rather than silently losing the W&B upload.
- The evaluation mode defaults to disabled even if training was enabled; shell wrappers can pick up the planned mode.
