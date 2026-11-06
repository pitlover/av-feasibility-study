"""Cosmos3-Edge AV inverse-dynamics SFT for cosmos-framework revision 98c35eda.

Export reads policy metadata from
  dataloader_train.dataloaders.action_data.dataloader.dataset.list_of_datasets
while this framework's SFT trainer uses a PackingDataLoader with a
RankPartitionedDataLoader child. Both routes are backed by ONE dataset config:
we don't insert a metadata-only copy that the trainer never consumes.

This is a custom/experimental AV SFT recipe, NOT an NVIDIA published AV
fine-tuning recipe. It retains the Edge model and the previously working
training hyperparameters from cosmos3av.
"""
from __future__ import annotations

import copy

EXPERIMENT_NAME = "action_inverse_dynamics_av_edge"
AV_ACTION_STEPS = 60
AV_FPS = 10
AV_DOMAIN = "av"


def _model_name_from_config(model_cfg):
    return str(model_cfg["vlm_config"]["model_instance"]["_target_"])


def get_av_dataset_with_policy_metadata(
    *,
    manifest: str,
    chunk_length: int = AV_ACTION_STEPS,
    fps: float = AV_FPS,
    embodiment_type: str = AV_DOMAIN,
    **dataset_kwargs,
):
    """The actual dataset factory (the very same leaf read by Edge export).

    The extra fields describe the samples and are validated rather than
    silently discarded. The underlying dataset still performs the original
    ActionTransformPipeline transformation.
    """
    if chunk_length != AV_ACTION_STEPS or fps != AV_FPS or embodiment_type != AV_DOMAIN:
        raise ValueError(
            "AV dataset contract is 61 video frames / 60 actions / 10 fps / domain 'av'; "
            f"got chunk_length={chunk_length}, fps={fps}, embodiment_type={embodiment_type}"
        )
    from .framework_dataset import get_action_av_sft_dataset
    return get_action_av_sft_dataset(manifest=manifest, **dataset_kwargs)


def make_av_packing_dataloader(*, dataloaders, rank_loader_kwargs, packing_kwargs):
    """Adapt the actual AV dataset leaf into the installed SFT packing loader.

    `dataloaders` is NOT a second/synthetic metadata tree. We select its
    canonical action_data dataset node and pass it to RankPartitionedDataLoader.
    This allows the runtime trainer and official Edge exporter to refer to
    exactly the same dataset configuration.
    """
    from cosmos_framework.data.generator.joint_dataloader import (
        PackingDataLoader, RankPartitionedDataLoader,
    )

    try:
        entries = dataloaders["action_data"]["dataloader"]["dataset"]["list_of_datasets"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Missing canonical dataloaders.action_data dataset") from exc
    if len(entries) != 1 or "dataset" not in entries[0]:
        raise ValueError("Expected exactly one action dataset node")

    # RankPartitionedDataLoader invokes cosmos_framework.utils.lazy_config.instantiate
    # on this dataset configuration, after distributed rank is initialized.
    actual_dataset_config = entries[0]["dataset"]
    rank_loader = RankPartitionedDataLoader(
        datasets={"av": {"ratio": 1, "dataset": actual_dataset_config}},
        **dict(rank_loader_kwargs),
    )
    return PackingDataLoader(dataloader=rank_loader, **dict(packing_kwargs))


def _verify_contract(cfg):
    """CPU-only config inspection; must run before distributed GPU launch."""
    model_cfg = cfg["model"]["config"]
    target = _model_name_from_config(model_cfg)
    if "Nemotron3DenseVLTextForCausalLM" not in target:
        raise RuntimeError(f"Expected Cosmos3-Edge Nemotron backbone, got {target}")
    if not bool(model_cfg["action_gen"]):
        raise RuntimeError("Edge action_gen must be enabled")

    dl = cfg["dataloader_train"]
    if "make_av_packing_dataloader" not in str(dl["_target_"]):
        raise RuntimeError("Actual training loader is not the export-aware adapter")
    leaf = dl["dataloaders"]["action_data"]["dataloader"]["dataset"]["list_of_datasets"][0]["dataset"]
    for field, expected in (
        ("chunk_length", AV_ACTION_STEPS),
        ("fps", AV_FPS),
        ("embodiment_type", AV_DOMAIN),
    ):
        if leaf.get(field) != expected:
            raise RuntimeError(f"Invalid {field}: got {leaf.get(field)!r}, expected {expected!r}")
    if "get_av_dataset_with_policy_metadata" not in str(leaf["_target_"]):
        raise RuntimeError("AV loader and exporter are not using the same canonical dataset factory")
    if "LIBERO_ROOT" in repr(dl):
        raise RuntimeError("Stale LIBERO_ROOT reference remains in training loader")
    return leaf


def register():
    from hydra.core.config_store import ConfigStore
    from cosmos_framework.utils.lazy_config import LazyCall as L
    from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
    from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_libero_nano import (
        action_policy_libero_nano,
    )

    # Keep model and optimizer recipe that already completed a GPU smoke run.
    # We only replace the AV dataloader with an export-compatible adapter.
    cfg = copy.deepcopy(action_policy_libero_nano)
    model_cfg = copy.deepcopy(EDGE_MODEL_CONFIG)
    model_cfg["action_gen"] = True
    model_cfg["max_num_tokens_after_packing"] = 45056
    model_cfg["diffusion_expert_config"]["load_weights_from_pretrained"] = True
    cfg["model"] = {"config": model_cfg}

    cfg["job"] = copy.deepcopy(cfg["job"])
    cfg["job"]["name"] = EXPERIMENT_NAME
    cfg["job"]["group"] = "av_inverse_dynamics"

    opt = copy.deepcopy(cfg["optimizer"])
    opt["keys_to_select"] = list(dict.fromkeys(
        list(opt["keys_to_select"]) + ["k_norm_und_for_gen"]
    ))
    cfg["optimizer"] = opt

    old_train = copy.deepcopy(cfg["dataloader_train"])
    old_rank = copy.deepcopy(old_train["dataloader"])
    old_rank["num_workers"] = 2
    old_rank["prefetch_factor"] = 2
    # Retain previously successful packing and rank-loader parameters.
    rank_loader_kwargs = {k: copy.deepcopy(v) for k, v in old_rank.items()
                          if k not in ("_target_", "datasets")}
    packing_kwargs = {k: copy.deepcopy(v) for k, v in old_train.items()
                      if k not in ("_target_", "dataloader")}
    packing_kwargs["dataset_name"] = "action_libero"  # routing used in working smoke run
    packing_kwargs["max_samples_per_batch"] = 1

    # The dataset factory is configured ONCE here. During training the adapter
    # consumes this exact node; during export NVIDIA reads the same node.
    canonical_dataset = L(get_av_dataset_with_policy_metadata)(
        manifest="${oc.env:COSMOS_AV_MANIFEST}",
        chunk_length=AV_ACTION_STEPS,
        fps=AV_FPS,
        embodiment_type=AV_DOMAIN,
        resolution="480",
        max_action_dim="${model.config.max_action_dim}",
        tokenizer_config="${model.config.vlm_config.tokenizer}",
        cfg_dropout_rate=0.1,
        format_prompt_as_json=True,
        iterable_shuffle=True,
        episode_shuffle_seed=42,
    )
    cfg["dataloader_train"] = L(make_av_packing_dataloader)(
        dataloaders={"action_data": {"dataloader": {"dataset": {
            "list_of_datasets": [{"dataset": canonical_dataset}]
        }}}},
        rank_loader_kwargs=rank_loader_kwargs,
        packing_kwargs=packing_kwargs,
    )

    ckpt = copy.deepcopy(cfg["checkpoint"])
    ckpt["keys_to_skip_loading"] = ["net_ema."]
    ckpt["strict_resume"] = True
    ckpt["load_training_state"] = False
    cfg["checkpoint"] = ckpt

    _verify_contract(cfg)
    ConfigStore.instance().store(
        group="experiment", package="_global_", name=EXPERIMENT_NAME, node=cfg,
    )
    return cfg
