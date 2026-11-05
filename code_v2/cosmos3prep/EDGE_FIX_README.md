# Cosmos3-Edge AV SFT config patch (2026-11-05)

This patch corrects a prior experimental package which mixed Edge DCP with Nano/Qwen model configuration and inherited the LIBERO dataset environment variable.

* `cosmos3av/experiment.py` now replaces **complete top-level** model and dataloader configs and fails fast if Edge is not actually Nemotron or if `LIBERO_ROOT` remains.
* `--datasets kitti`, `--datasets waymo`, `--datasets kitti waymo` remain controlled by the existing `plan` / manifest.
* **No data regeneration, checkpoint reconversion, or new plan is required just to check the config.**
* This package remains **experimental and GPU training unverified**. Do not infer correct pose convention or action normalization from config success.

## Apply

From the project root `cosmos3_ji`, back up the current experiment code and copy the patched file:

```bash
cp cosmos3av/experiment.py cosmos3av/experiment.py.bak
cp /PATH_TO_UNZIPPED/cosmos3_edge_av_stage1/cosmos3av/experiment.py cosmos3av/experiment.py
```

## Preflight

```bash
python - <<'PY'
from cosmos3av.experiment import register
cfg = register()
name = cfg['model']['config']['vlm_config']['model_instance']['_target_']
print('Edge model:', name)
print('Dataloader datasets:', list(cfg['dataloader_train']['dataloader']['datasets']))
assert 'Nemotron3DenseVLTextForCausalLM' in str(name)
assert list(cfg['dataloader_train']['dataloader']['datasets']) == ['av']
PY
```

After this passes, run the existing 10-step `launch ... --execute` and inspect the *next* error (if any). Successful config checks do not establish GPU training compatibility.

If the **imported** `EDGE_MODEL_CONFIG` itself resolves to Qwen, your installed/repo `cosmos-framework` version may differ from NVIDIA's current official source. Do not force-load mismatched DCP. Inspect `cosmos_framework.configs.base.experiment.sft.models.edge_model_config.__file__` and align the official checkout/checkpoint versions first.
