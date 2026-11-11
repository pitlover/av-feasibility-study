# Stage1 TOML Configuration

This patch replaces **only** `stage1_full_train_eval.sh` and adds `scripts/stage1_toml_config.py` and `configs/stage1_full.toml`. It expects the previously integrated Cosmos3 W&B patch (`cosmos3av.cli --wandb-mode`, `cosmos3eval.cli score --wandb-mode`). It does **not** modify `cosmos3eval/camera_render.py`.

## Install

From `cosmos3_ji`, back up the existing shell script, then copy the three paths from the ZIP into the project root. Python 3.11+ is required for standard-library `tomllib`; the user's `cosmos` environment uses Python 3.13.

```bash
cp stage1_full_train_eval.sh stage1_full_train_eval.sh.bak
unzip -o cosmos3_stage1_toml_patch.zip
bash -n stage1_full_train_eval.sh
```

## Launch

```bash
bash stage1_full_train_eval.sh --toml configs/stage1_full.toml config
bash stage1_full_train_eval.sh --toml configs/stage1_full.toml inspect
bash stage1_full_train_eval.sh --toml configs/stage1_full.toml plan
bash stage1_full_train_eval.sh --toml configs/stage1_full.toml train
bash stage1_full_train_eval.sh --toml configs/stage1_full.toml export
bash stage1_full_train_eval.sh --toml configs/stage1_full.toml evaluate
```

You can also run `all` from an unused `job.run_tag`. The stages are intentionally separated to allow reviewing results and failures.

## CLI overrides

```bash
bash stage1_full_train_eval.sh --toml configs/stage1_full.toml \
  train.max_iters=500 train.save_every=100 job.run_tag=s1_test500 plan
```

Overrides apply **only** to the command where they are specified. To run `train`, use the **same TOML/overrides** as `plan` (or change the TOML file before planning). The script checks that planned training settings have not changed. `export.step` and `[eval]` settings can change after training without retraining. If `[export] step` is set to 3000 in the TOML, overriding `train.max_iters` alone does not change the export step; set `export.step=0` to track `train.max_iters` automatically.

## W&B examples

```toml
[wandb]
enabled = true
mode = "online"   # or "offline"; disabled is default
```

With `enabled=false`, the mode must be `disabled`. `online` requires `pip install wandb` and login. The script sends `wandb.mode` to `cosmos3av.cli plan/launch` and to `cosmos3eval.cli score`. **W&B metrics are logged at evaluation time**; this does not run evaluation during every training step.

## Supported settings

- `job.run_tag`, `job.eval_tag`, `job.run_step` (step used when none is given on the command line, e.g. by the no-argument entrypoints `configs/scripts/run_stage1_reptile_*.sh`; default `config`)
- `paths.base`, `paths.framework`, `paths.processed`, `paths.template_run`, `paths.edge_dcp`, `paths.wan_vae`
- `paths.runs_root`, `paths.exports_root`, `paths.evaluation_root` (save destinations; relative = under `paths.base`; defaults `runs_stage1_full`, `exports`, `evaluation` = the unchanged layout)
- `data.epoch_clips`, `data.seed`, `data.weights.kitti`, `data.weights.waymo`
- `train.gpus`, `train.max_iters`, `train.save_every`
- `optimizer.lr`
- `wandb.enabled`, `wandb.mode`
- `export.step` (0 = final training checkpoint)
- `eval.clips_per_dataset`, `eval.infer_gpus`, `eval.ground_height_m`, `eval.horizon_steps`, `eval.fps`, `eval.layout`

Only these keys are supported. Unlike OpenPI, this Cosmos3 training adapter does **not** currently expose `train.batch_size`, `optimizer.weight_decay`, `optimizer.clip_gradient_norm`, `ema_enabled`, or other advanced framework parameters through this wrapper. Such keys fail explicitly; they are **not** silently ignored. Those settings require separate work on the underlying Cosmos Framework recipe.

Keep the same `job.run_tag` and config file across `plan/train/export/evaluate`. Use a new `job.run_tag` for each new training experiment. Use a new `job.eval_tag` to rerun inference without overwriting previous predictions. A copy of the TOML and resolved JSON is saved in the run directory at `plan` time.
