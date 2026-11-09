# Stage2 TOML + W&B support

This patch adds TOML-driven configuration and W&B support to the Stage2 repro pipeline.

## Files

- `retrain_cosmos3_av_repro_v3.sh`
- `scripts/stage2_toml_config.py`
- `configs/stage2_repro.toml`

## What it supports

- Stage1 pretrain (`stage1`) and Stage2 posttrain (`stage2`) planning/training from one TOML
- `wandb.mode = disabled | online | offline`
- Evaluation logging for both Stage1 and Stage2 via `cosmos3eval.cli score`
- CLI overrides with `section.key=value`

## Commands

```bash
export CONFIG="configs/stage2_repro.toml"

bash retrain_cosmos3_av_repro_v3.sh --toml "$CONFIG" config
bash retrain_cosmos3_av_repro_v3.sh --toml "$CONFIG" stage1
bash retrain_cosmos3_av_repro_v3.sh --toml "$CONFIG" stage2
bash retrain_cosmos3_av_repro_v3.sh --toml "$CONFIG" export
bash retrain_cosmos3_av_repro_v3.sh --toml "$CONFIG" evaluate
```

Or all at once:

```bash
bash retrain_cosmos3_av_repro_v3.sh --toml "$CONFIG" all
```

## Notes

- `paths.edge_dcp` and `paths.wan_vae` are supported directly in the TOML.
- Stage2 initializes from the Stage1 checkpoint produced by the same `run_tag` and `max_iters`.
- Evaluation logs to W&B only when `wandb.mode` is `online` or `offline`.
- Oracle score is intentionally not logged to W&B.

## Example CLI override

```bash
bash retrain_cosmos3_av_repro_v3.sh \
  --toml configs/stage2_repro.toml \
  stage2 \
  job.run_tag=s2_repro_v2 \
  train.max_iters=500 \
  wandb.mode="online"
```
