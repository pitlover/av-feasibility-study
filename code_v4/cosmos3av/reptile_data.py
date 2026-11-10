"""Rank-synchronized clip stream for the Reptile trainer, served through the baseline's own loader stack.

The joint Stage 1 baseline feeds ``OmniMoTModel.training_step`` with
``PackingDataLoader(RankPartitionedDataLoader(DataLoader(dataset)))``. The Reptile trainer keeps exactly
that stack -- same collate, same packing, same batch format -- and only replaces the dataset by
:class:`ReptileClipStreamDataset`, an iterable dataset that yields the clips of one rank in the order
``ReptileEpisodeSampler.plan_for_rank`` prescribes (episode after episode: a task's query clips, then one
batch of ``clips_per_rank`` clips per inner step). Every yielded sample carries ``_reptile_*`` integer
keys so the trainer can assert that the batch it pulls is the one its schedule expects.

DataLoader workers: worker ``w`` of ``nw`` yields the stream positions ``p == w (mod nw)``; with
``in_order=True`` the DataLoader interleaves its workers round-robin, so the consumer sees the stream in
order (the baseline's ``in_order=False`` is overridden for this reason).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from torch.utils.data import IterableDataset, get_worker_info

from .reptile import ClipTables, ReptileConfig, ReptileEpisodeSampler, StreamItem, load_manifest_rows, unique_rows

META_KEYS = ("_reptile_spec_id", "_reptile_task", "_reptile_role", "_reptile_step", "_reptile_slot", "_reptile_clip_index")

# keys of the canonical AV dataset node that are NOT arguments of get_action_av_sft_dataset
_NODE_KEYS_DROPPED = ("_target_", "chunk_length", "fps", "embodiment_type", "manifest", "iterable_shuffle", "episode_shuffle_seed")


def item_meta(item: StreamItem) -> dict[str, int]:
    return {
        "_reptile_spec_id": int(item.spec_id),
        "_reptile_task": int(item.task),
        "_reptile_role": int(item.role),
        "_reptile_step": int(item.step),
        "_reptile_slot": int(item.slot),
        "_reptile_clip_index": int(item.clip_index),
    }


class ReptileClipStreamDataset(IterableDataset):
    """Infinite per-rank clip stream in Reptile episode order (one transformed sample per item).

    ``shard_rank`` / ``shard_world_size`` are set by ``RankPartitionedDataLoader`` (every rank belongs to the
    single ``av`` partition, so ``shard_world_size`` == the world size the sampler was built with).
    """

    def __init__(self, dataset: Any, sampler: ReptileEpisodeSampler, spec_offset: int = 0) -> None:
        super().__init__()
        self._dataset = dataset  # ActionSFTDataset over the unique-clip manifest (map-style)
        self.sampler = sampler
        self.spec_offset = int(spec_offset)
        self.shard_rank = 0
        self.shard_world_size = 1
        self.shard_id = 0

    def __len__(self) -> int:  # informational only; iteration is infinite
        return len(self._dataset)

    def load_item(self, item: StreamItem) -> dict[str, Any]:
        sample = self._dataset[int(item.clip_index)]
        sample.update(item_meta(item))
        return sample

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if int(self.shard_world_size) != int(self.sampler.world_size):
            raise RuntimeError(
                f"RankPartitionedDataLoader gave this dataset shard_world_size={self.shard_world_size} but the sampler was "
                f"built for world_size={self.sampler.world_size}; the Reptile loader must own all ranks"
            )
        wi = get_worker_info()
        worker_id = wi.id if wi is not None else 0
        num_workers = wi.num_workers if wi is not None else 1
        rank = int(self.shard_rank)
        position = 0
        spec_id = self.spec_offset
        while True:
            spec = self.sampler.spec(spec_id)
            for item in self.sampler.plan_for_rank(spec, rank):
                if position % num_workers == worker_id:
                    yield self.load_item(item)
                position += 1
            spec_id += 1


def dataset_kwargs_from_node(node: Any) -> dict[str, Any]:
    """Arguments of ``get_action_av_sft_dataset`` from the experiment's canonical AV dataset node (resolved),
    minus the contract fields validated by ``get_av_dataset_with_policy_metadata`` and the shuffle knobs."""
    from omegaconf import OmegaConf

    data = OmegaConf.to_container(node, resolve=True) if OmegaConf.is_config(node) else dict(node)
    return {k: v for k, v in dict(data).items() if k not in _NODE_KEYS_DROPPED}


def build_reptile_stream_dataset(
    *,
    reptile_config_path: str,
    world_size: int,
    spec_offset: int = 0,
    **dataset_kwargs: Any,
) -> ReptileClipStreamDataset:
    """``LazyCall`` target instantiated by ``RankPartitionedDataLoader`` on every rank.

    Builds the AV action dataset (``cosmos3av.framework_dataset.get_action_av_sft_dataset`` over the unique-clip
    manifest recorded in the Reptile config, map-style) and the episode sampler, and wraps them in the stream.
    """
    from .framework_dataset import get_action_av_sft_dataset

    cfg = ReptileConfig.load(reptile_config_path)
    if not cfg.manifest or not Path(cfg.manifest).is_file():
        raise FileNotFoundError(f"Reptile config has no usable unique-clip manifest: {cfg.manifest!r}")
    tables = ClipTables(unique_rows(load_manifest_rows(cfg.manifest)))
    sampler = ReptileEpisodeSampler(tables, cfg, int(world_size))
    dataset = get_action_av_sft_dataset(manifest=cfg.manifest, iterable_shuffle=False, **dataset_kwargs)
    if len(dataset) != len(tables):
        raise RuntimeError(f"manifest/table size mismatch: dataset {len(dataset)} vs tables {len(tables)}")
    return ReptileClipStreamDataset(dataset, sampler, spec_offset=int(spec_offset))


def build_reptile_loader(config: Any, reptile_config_path: str, world_size: int, spec_offset: int) -> Any:
    """The baseline's ``PackingDataLoader(RankPartitionedDataLoader(...))`` around :class:`ReptileClipStreamDataset`.

    ``config.dataloader_train`` is the experiment's ``make_av_packing_dataloader`` node; its ``rank_loader_kwargs`` /
    ``packing_kwargs`` are reused verbatim except ``in_order=True`` (deterministic worker interleaving) and
    ``max_samples_per_batch=clips_per_rank`` (one inner step per packed batch).
    """
    from omegaconf import OmegaConf

    from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, RankPartitionedDataLoader
    from cosmos_framework.utils.lazy_config import LazyCall as L

    cfg = ReptileConfig.load(reptile_config_path)
    dl = config.dataloader_train
    node = dl["dataloaders"]["action_data"]["dataloader"]["dataset"]["list_of_datasets"][0]["dataset"]
    dataset_kwargs = dataset_kwargs_from_node(node)
    rank_kwargs = dict(OmegaConf.to_container(dl["rank_loader_kwargs"], resolve=True))
    packing_kwargs = dict(OmegaConf.to_container(dl["packing_kwargs"], resolve=True))
    rank_kwargs["batch_size"] = 1
    rank_kwargs["in_order"] = True
    rank_kwargs.pop("sampler", None)
    packing_kwargs["max_samples_per_batch"] = int(cfg.clips_per_rank)
    packing_kwargs["max_sequence_length"] = None
    dataset_cfg = L(build_reptile_stream_dataset)(
        reptile_config_path=str(reptile_config_path),
        world_size=int(world_size),
        spec_offset=int(spec_offset),
        **dataset_kwargs,
    )
    rank_loader = RankPartitionedDataLoader(datasets={"av": {"ratio": 1, "dataset": dataset_cfg}}, **rank_kwargs)
    return PackingDataLoader(dataloader=rank_loader, **packing_kwargs)


__all__ = [
    "META_KEYS",
    "ReptileClipStreamDataset",
    "build_reptile_loader",
    "build_reptile_stream_dataset",
    "dataset_kwargs_from_node",
    "item_meta",
]
