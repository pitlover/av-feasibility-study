"""EXPERIMENTAL Cosmos3-Edge AV Inverse Dynamics SFT experiment.

Avoids in-place edits of nested Nano/LIBERO LazyDict configs. Copies/replaces
entire top-level model and dataloader entries to avoid stale configurations.

This is NOT an NVIDIA-validated AV inverse-dynamics fine-tuning recipe.
"""
from __future__ import annotations

import copy


def _model_name_from_config(model_cfg):
    return str(model_cfg["vlm_config"]["model_instance"]["_target_"])


def register():
    from hydra.core.config_store import ConfigStore
    from cosmos_framework.utils.lazy_config import LazyCall as L
    from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
    from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_libero_nano import (
        action_policy_libero_nano,
    )
    from .framework_dataset import get_action_av_sft_dataset

    model_cfg = copy.deepcopy(EDGE_MODEL_CONFIG)
    model_name = _model_name_from_config(model_cfg)
    if "Nemotron3DenseVLTextForCausalLM" not in model_name:
        raise RuntimeError(
            "The imported EDGE_MODEL_CONFIG is not the Edge/Nemotron model: "
            f"{model_name}. Check the cosmos-framework checkout/version in use "
            "and do NOT launch training until it resolves to the Edge backbone."
        )

    model_cfg["action_gen"] = True
    model_cfg["max_num_tokens_after_packing"] = 45056
    model_cfg["diffusion_expert_config"]["load_weights_from_pretrained"] = True

    # Retain action-SFT recipe's optimizer/trainer mechanics, but REPLACE the
    # entire model subtree rather than mutating nested LazyDict views.
    cfg = copy.deepcopy(action_policy_libero_nano)
    job = copy.deepcopy(cfg["job"])
    job["name"] = "action_inverse_dynamics_av_edge"
    job["group"] = "av_inverse_dynamics"
    cfg["job"] = job
    cfg["model"] = {"config": model_cfg}

    opt = copy.deepcopy(cfg["optimizer"])
    opt["keys_to_select"] = list(dict.fromkeys(
        list(opt["keys_to_select"]) + ["k_norm_und_for_gen"]
    ))
    cfg["optimizer"] = opt

    # Build the AV dataloader from scratch. The original LIBERO-specific
    # dataset subtree and its ${oc.env:LIBERO_ROOT} must NOT survive.
    dataloader_train = copy.deepcopy(cfg["dataloader_train"])
    rank_loader = copy.deepcopy(dataloader_train["dataloader"])
    rank_loader["num_workers"] = 2
    rank_loader["prefetch_factor"] = 2
    rank_loader["datasets"] = {
        "av": {
            "ratio": 1,
            "dataset": L(get_action_av_sft_dataset)(
                manifest="${oc.env:COSMOS_AV_MANIFEST}",
                resolution="480",
                max_action_dim="${model.config.max_action_dim}",
                tokenizer_config="${model.config.vlm_config.tokenizer}",
                cfg_dropout_rate=0.1,
                format_prompt_as_json=True,
                iterable_shuffle=True,
                episode_shuffle_seed=42,
            ),
        },
    }
    dataloader_train["dataset_name"] = "action_libero"  # Preserve official action collator routing
    dataloader_train["max_samples_per_batch"] = 1
    dataloader_train["dataloader"] = rank_loader
    cfg["dataloader_train"] = dataloader_train

    ckpt = copy.deepcopy(cfg["checkpoint"])
    ckpt["keys_to_skip_loading"] = ["net_ema."]
    ckpt["strict_resume"] = True
    ckpt["load_training_state"] = False
    cfg["checkpoint"] = ckpt

    # Explicit fail-fast checks. These must pass BEFORE constructing GPUs.
    resolved_model = cfg["model"]["config"]
    resolved_name = _model_name_from_config(resolved_model)
    if "Nemotron3DenseVLTextForCausalLM" not in resolved_name:
        raise RuntimeError(f"Edge model replacement did not stick: {resolved_name}")
    datasets_text = repr(cfg["dataloader_train"])
    if "LIBERO_ROOT" in datasets_text or "get_action_libero_sft_dataset" in datasets_text:
        raise RuntimeError("Stale LIBERO dataset configuration remains in dataloader_train")

    ConfigStore.instance().store(
        group="experiment", package="_global_",
        name="action_inverse_dynamics_av_edge", node=cfg,
    )
    return cfg
