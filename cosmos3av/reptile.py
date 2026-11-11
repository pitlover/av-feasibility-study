"""Reptile meta-pretraining for Stage 1 (KITTI + Waymo): configuration and episode sampling.

Reptile (Nichol, Achiam & Schulman 2018) replaces joint training with a two-level loop::

    for meta-iteration j:
        for every task t of the meta-iteration (one dataset, or both -- see ``episode_mode``):
            theta_tilde_t = k ordinary optimizer steps on clips of t, starting from theta
        theta <- theta + eps(j) * mean_t(theta_tilde_t - theta)

The *task* of the autonomous-driving setting is a dataset (sensor rig, geography, speed profile);
the downstream Stage 2 post-training on Argoverse 2 is "adapt to a new dataset", so the inner loop
is made to look like it: the ordinary Stage 1 optimizer (same FusedAdam, learning rate, multipliers,
gradient clipping), a fresh mini-batch of clips per step and a fresh optimizer state per task.

This module is numpy-only (no torch) so that ``cosmos3av.cli plan`` can validate a Reptile config and
the sampler can be unit-tested without the Cosmos framework. The torch side lives in
``cosmos3av.reptile_meta`` (theta bookkeeping) and ``cosmos3av.reptile_data`` / ``reptile_train``.

Episode modes (``ReptileConfig.episode_mode``):

``per_dataset``
    one dataset per meta-iteration (serial Reptile). ``dataset_schedule`` picks it: ``alternate``
    cycles through the datasets in order (kitti, waymo, kitti, ...), ``random`` draws it from
    ``dataset_weights``.
``all_datasets``
    every dataset gets its own inner loop in every meta-iteration, all starting from the same theta;
    the displacements are averaged with ``dataset_weights`` before the meta step (batched Reptile).
``mixed``
    one inner loop per meta-iteration whose mini-batches mix the datasets in ``dataset_weights``
    proportions -- the sampling of the joint baseline inside the Reptile outer loop (a control that
    has no per-dataset task structure).

Per-rank data layout: all ranks run the same inner loop on the same task (FSDP needs identical
weights), so every rank derives the *same* episode from ``default_rng([seed, j])`` and takes its own
shard of the clips. One inner step consumes ``clips_per_rank`` clips on every rank, i.e.
``world_size * clips_per_rank`` clips globally -- the same global batch as the joint baseline when
``clips_per_rank`` equals the baseline's ``max_samples_per_batch``.
"""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

EPISODE_MODES = ("per_dataset", "all_datasets", "mixed")
DATASET_SCHEDULES = ("alternate", "random")
LOSS_MODES = ("total", "action")
ROLE_QUERY = 0
ROLE_SUPPORT = 1
MIXED_NAME = "mixed"


# --------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------
@dataclass
class ReptileConfig:
    """Knobs of the Reptile trainer. ``datasets`` / ``dataset_weights`` / ``manifest`` are filled by
    ``cosmos3av.cli plan`` from its own arguments; everything else comes from the ``[reptile]`` table
    of the Stage 1 TOML (``scripts/stage1_toml_config.py --format reptile-json``)."""

    episode_mode: str = "per_dataset"
    dataset_schedule: str = "alternate"
    # inner loop = k steps of the Stage 1 optimizer on one task
    inner_steps: int = 10
    inner_warmup_steps: int = 2  # linear LR ramp over the first steps of every inner loop
    inner_grad_clip: float | None = 1.0  # None / <= 0 disables (the baseline clips at 1.0)
    inner_reset_optimizer: bool = True  # fresh Adam moments per task (each inner loop = a fresh fine-tune)
    # outer loop: theta <- theta + eps * d; eps anneals linearly from meta_lr to meta_lr * meta_lr_min_ratio at max_iter
    meta_lr: float = 0.5
    meta_lr_min_ratio: float = 0.1
    meta_lr_warmup_iters: int = 0
    # data per inner step / per episode
    clips_per_rank: int = 1  # clips per rank per inner step (== the baseline's max_samples_per_batch)
    support_clips: int = 0  # 0 = a fresh batch every inner step (inner_steps * world * clips_per_rank clips); >0 = fixed set, cycled
    query_clips_per_rank: int = 1  # held-out clips per rank for the zero-shot vs adapted diagnostic
    query_sequence_disjoint: bool = True  # query clips from drives (sequences) not used in the support set when possible
    eval_query_every: int = 10  # 0 = never
    loss_mode: str = "total"  # "total" = the recipe objective (vision + action flow matching), "action" = action term only
    seed: int = 42
    # filled by plan
    datasets: list[str] = field(default_factory=list)
    dataset_weights: dict[str, float] = field(default_factory=dict)
    manifest: str = ""  # unique-clip train manifest written by plan

    # ---- validation / (de)serialization ------------------------------------------------------
    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.episode_mode not in EPISODE_MODES:
            raise ValueError(f"episode_mode must be one of {EPISODE_MODES}, got {self.episode_mode!r}")
        if self.dataset_schedule not in DATASET_SCHEDULES:
            raise ValueError(f"dataset_schedule must be one of {DATASET_SCHEDULES}, got {self.dataset_schedule!r}")
        if self.loss_mode not in LOSS_MODES:
            raise ValueError(f"loss_mode must be one of {LOSS_MODES}, got {self.loss_mode!r}")
        if int(self.inner_steps) < 1:
            raise ValueError("inner_steps must be >= 1")
        if int(self.inner_warmup_steps) < 0 or int(self.inner_warmup_steps) > int(self.inner_steps):
            raise ValueError("inner_warmup_steps must be in [0, inner_steps]")
        if self.inner_grad_clip is not None and float(self.inner_grad_clip) <= 0:
            self.inner_grad_clip = None
        if not (0.0 < float(self.meta_lr) <= 1.0):
            raise ValueError("meta_lr (Reptile eps) must be in (0, 1]")
        if not (0.0 < float(self.meta_lr_min_ratio) <= 1.0):
            raise ValueError("meta_lr_min_ratio must be in (0, 1]")
        if int(self.meta_lr_warmup_iters) < 0:
            raise ValueError("meta_lr_warmup_iters must be >= 0")
        if int(self.clips_per_rank) < 1:
            raise ValueError("clips_per_rank must be >= 1")
        if int(self.support_clips) < 0:
            raise ValueError("support_clips must be >= 0 (0 = fresh clips every inner step)")
        if int(self.query_clips_per_rank) < 1:
            raise ValueError("query_clips_per_rank must be >= 1")
        if int(self.query_clips_per_rank) % int(self.clips_per_rank) != 0:
            # the packing loader groups clips_per_rank consecutive clips; a role change must fall on a batch boundary
            raise ValueError("query_clips_per_rank must be a multiple of clips_per_rank")
        if int(self.eval_query_every) < 0:
            raise ValueError("eval_query_every must be >= 0")
        if self.datasets:
            if len(set(self.datasets)) != len(self.datasets):
                raise ValueError("datasets must be unique")
            if set(self.dataset_weights) != set(self.datasets):
                raise ValueError(f"dataset_weights must cover exactly {self.datasets}, got {sorted(self.dataset_weights)}")
            for name, w in self.dataset_weights.items():
                if not math.isfinite(float(w)) or float(w) <= 0:
                    raise ValueError(f"dataset_weights[{name!r}] must be positive and finite")
            if self.episode_mode == "per_dataset" and self.dataset_schedule == "alternate":
                ws = [float(self.dataset_weights[n]) for n in self.datasets]
                if any(abs(w - ws[0]) > 1e-9 for w in ws):
                    raise ValueError(
                        "dataset_schedule='alternate' visits every dataset equally often; use equal data.weights or "
                        "dataset_schedule='random'"
                    )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ReptileConfig":
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(raw) - names)
        if unknown:
            raise ValueError(f"Unknown Reptile config key(s) {unknown}; valid: {sorted(names)}")
        kwargs = {k: _plain(v) for k, v in raw.items()}
        if "datasets" in kwargs:
            kwargs["datasets"] = [str(x) for x in kwargs["datasets"]]
        if "dataset_weights" in kwargs:
            kwargs["dataset_weights"] = {str(k): float(v) for k, v in dict(kwargs["dataset_weights"]).items()}
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def load(cls, path: str | Path) -> "ReptileConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf8")))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf8")
        return path

    # ---- derived sizes --------------------------------------------------------------------
    def support_size(self, world_size: int) -> int:
        """Clips of one task per meta-iteration (all ranks together)."""
        fresh = int(self.inner_steps) * int(world_size) * int(self.clips_per_rank)
        return int(self.support_clips) if int(self.support_clips) > 0 else fresh

    def query_size(self, world_size: int) -> int:
        return int(world_size) * int(self.query_clips_per_rank)

    def items_per_rank_per_task(self) -> int:
        """Clips every rank consumes per task: the query clips first, then one batch per inner step."""
        return int(self.query_clips_per_rank) + int(self.inner_steps) * int(self.clips_per_rank)

    def normalized_weights(self) -> dict[str, float]:
        total = float(sum(float(self.dataset_weights[n]) for n in self.datasets))
        return {n: float(self.dataset_weights[n]) / total for n in self.datasets}


def _plain(value: Any) -> Any:
    """OmegaConf containers -> plain Python (the config may travel through a LazyCall)."""
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(value, (DictConfig, ListConfig)):
            return OmegaConf.to_container(value, resolve=True)
    except ImportError:  # pragma: no cover - omegaconf is always present on the training side
        pass
    return value


def eps_at(cfg: ReptileConfig, iteration: int, max_iter: int) -> float:
    """Reptile step size at ``iteration``: optional linear warm-up, then linear decay from ``meta_lr``
    to ``meta_lr * meta_lr_min_ratio`` reached exactly at ``max_iter``."""
    warm = max(0, int(cfg.meta_lr_warmup_iters))
    if warm > 0 and iteration < warm:
        return float(cfg.meta_lr) * float(iteration + 1) / warm
    span = max(1, int(max_iter) - warm)
    progress = min(1.0, max(0.0, (iteration - warm) / span))
    return float(cfg.meta_lr) * (1.0 - (1.0 - float(cfg.meta_lr_min_ratio)) * progress)


# --------------------------------------------------------------------------------------------
# Clip tables (from the train manifest)
# --------------------------------------------------------------------------------------------
def load_manifest_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def unique_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The manifest of ``plan`` repeats clips to fill ``epoch_clips``; Reptile samples clips itself and needs
    every clip exactly once. Sorted by ``clip_id`` so the table is identical on every rank and worker."""
    seen: dict[str, dict[str, Any]] = {}
    for r in rows:
        cid = str(r["clip_id"])
        if cid not in seen:
            seen[cid] = dict(r)
    return [seen[k] for k in sorted(seen)]


class ClipTables:
    """Index of the unique clips of a manifest, grouped by dataset (``rows[i]['dataset']``)."""

    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        ids = [str(r["clip_id"]) for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError("ClipTables needs unique clip_ids (use unique_rows())")
        self.rows: list[dict[str, Any]] = [dict(r) for r in rows]
        self.clip_ids: list[str] = ids
        self.datasets_of: list[str] = [str(r["dataset"]) for r in rows]
        self.sequences_of: list[str] = [str(r.get("sequence", r["clip_id"])) for r in rows]
        self.by_dataset: dict[str, np.ndarray] = {}
        for name in sorted(set(self.datasets_of)):
            self.by_dataset[name] = np.asarray([i for i, d in enumerate(self.datasets_of) if d == name], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.rows)

    def summary(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for name, idx in self.by_dataset.items():
            out[name] = {"clips": int(len(idx)), "sequences": len({self.sequences_of[int(i)] for i in idx})}
        return out


# --------------------------------------------------------------------------------------------
# Episodes
# --------------------------------------------------------------------------------------------
@dataclass
class TaskEpisode:
    """One inner loop: the task name, its displacement weight, and the clip indices (into ``ClipTables.rows``)
    in global consumption order."""

    name: str
    weight: float
    support: list[int]
    query: list[int]
    support_datasets: dict[str, int]  # how many support clips came from each dataset
    query_sequence_disjoint: bool  # True when every query clip is from a drive absent from the support set
    query_overlaps_support: bool  # True only for tiny datasets where disjoint sampling was impossible

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "weight": self.weight,
            "num_support": len(self.support),
            "num_query": len(self.query),
            "support_datasets": dict(self.support_datasets),
            "query_sequence_disjoint": self.query_sequence_disjoint,
            "query_overlaps_support": self.query_overlaps_support,
        }


@dataclass
class EpisodeSpec:
    spec_id: int
    tasks: list[TaskEpisode]

    def summary(self) -> dict[str, Any]:
        return {"spec_id": self.spec_id, "tasks": [t.summary() for t in self.tasks]}


@dataclass(frozen=True)
class StreamItem:
    """One clip of one rank's stream: which episode / task / role / inner step it belongs to."""

    spec_id: int
    task: int
    role: int  # ROLE_QUERY or ROLE_SUPPORT
    step: int  # inner step for support clips, -1 for query clips
    slot: int  # position inside the step (0..clips_per_rank-1) or inside the query set
    clip_index: int


def _draw(rng: np.random.Generator, pool: np.ndarray, count: int) -> list[int]:
    """``count`` clips from ``pool`` without replacement; when ``count`` exceeds the pool every clip is
    used once (random order) before any repeats."""
    n = int(len(pool))
    if n == 0:
        raise ValueError("cannot draw from an empty clip pool")
    if count <= n:
        return [int(x) for x in pool[rng.choice(n, size=count, replace=False)]]
    picks: list[int] = []
    while len(picks) < count:
        picks.extend(int(x) for x in pool[rng.permutation(n)])
    return picks[:count]


def _quotas(weights: Sequence[float], total: int) -> list[int]:
    """Largest-remainder split of ``total`` by ``weights`` (same rule as ``cosmos3av.data.mix_rows``)."""
    s = float(sum(weights))
    exact = [total * float(w) / s for w in weights]
    picks = [int(math.floor(x)) for x in exact]
    rest = total - sum(picks)
    order = sorted(range(len(weights)), key=lambda i: (exact[i] - picks[i], -i), reverse=True)
    for i in order[:rest]:
        picks[i] += 1
    return picks


class ReptileEpisodeSampler:
    """Deterministic, rank-synchronized episode specs (pure bookkeeping, no I/O).

    ``spec(j)`` is drawn from ``default_rng([seed, j])`` and therefore identical on every rank and
    DataLoader worker; ``plan_for_rank(spec, rank)`` is the slice of it one rank consumes, in order.
    """

    def __init__(self, tables: ClipTables, cfg: ReptileConfig, world_size: int) -> None:
        if not cfg.datasets:
            raise ValueError("ReptileConfig.datasets is empty -- run through cosmos3av.cli plan")
        missing = [d for d in cfg.datasets if d not in tables.by_dataset]
        if missing:
            raise ValueError(f"manifest has no clips for dataset(s) {missing}; available: {sorted(tables.by_dataset)}")
        if int(world_size) < 1:
            raise ValueError("world_size must be >= 1")
        self.tables = tables
        self.cfg = cfg
        self.world_size = int(world_size)
        self.datasets = list(cfg.datasets)
        self.weights = cfg.normalized_weights()
        self.support_size = cfg.support_size(self.world_size)
        self.query_size = cfg.query_size(self.world_size)
        if self.support_size < self.world_size * int(cfg.clips_per_rank):
            raise ValueError(
                f"support_clips={cfg.support_clips} < world_size * clips_per_rank = "
                f"{self.world_size * int(cfg.clips_per_rank)}: an inner step would reuse a clip on two ranks"
            )

    # ---- helpers -------------------------------------------------------------------------
    def task_names_for(self, spec_id: int, rng: np.random.Generator) -> list[str]:
        mode = self.cfg.episode_mode
        if mode == "all_datasets":
            return list(self.datasets)
        if mode == "mixed":
            return [MIXED_NAME]
        if self.cfg.dataset_schedule == "alternate":
            return [self.datasets[int(spec_id) % len(self.datasets)]]
        probs = np.asarray([self.weights[n] for n in self.datasets], dtype=np.float64)
        return [self.datasets[int(rng.choice(len(self.datasets), p=probs))]]

    def _pool(self, name: str) -> np.ndarray:
        return self.tables.by_dataset[name]

    def _draw_mixture(self, rng: np.random.Generator, count: int) -> list[int]:
        quotas = _quotas([self.weights[n] for n in self.datasets], count)
        picks: list[int] = []
        for name, q in zip(self.datasets, quotas):
            if q > 0:
                picks.extend(_draw(rng, self._pool(name), q))
        picks = [int(x) for x in np.asarray(picks, dtype=np.int64)[rng.permutation(len(picks))]]
        return picks

    def _draw_query(self, rng: np.random.Generator, pools: list[np.ndarray], support: list[int], count: int) -> tuple[list[int], bool, bool]:
        """Held-out clips: never in the support set and, when possible, from other drives (sequences)."""
        support_set = set(support)
        support_seqs = {self.tables.sequences_of[i] for i in support}
        remaining = np.asarray([int(i) for pool in pools for i in pool if int(i) not in support_set], dtype=np.int64)
        preferred = np.asarray([int(i) for i in remaining if self.tables.sequences_of[int(i)] not in support_seqs], dtype=np.int64)
        if self.cfg.query_sequence_disjoint and len(preferred) >= count:
            return _draw(rng, preferred, count), True, False
        if len(remaining) >= count:
            return _draw(rng, remaining, count), False, False
        # tiny dataset (e.g. a smoke run on a handful of clips): fall back to the whole pool
        whole = np.concatenate(pools) if len(pools) > 1 else pools[0]
        return _draw(rng, whole, count), False, True

    # ---- main entry --------------------------------------------------------------------
    def spec(self, spec_id: int) -> EpisodeSpec:
        rng = np.random.default_rng([int(self.cfg.seed), int(spec_id)])
        tasks: list[TaskEpisode] = []
        for name in self.task_names_for(spec_id, rng):
            if name == MIXED_NAME:
                support = self._draw_mixture(rng, self.support_size)
                pools = [self._pool(n) for n in self.datasets]
                weight = 1.0
            else:
                support = _draw(rng, self._pool(name), self.support_size)
                pools = [self._pool(name)]
                weight = self.weights[name] if self.cfg.episode_mode == "all_datasets" else 1.0
            query, seq_disjoint, overlap = self._draw_query(rng, pools, support, self.query_size)
            counts: dict[str, int] = {}
            for i in support:
                d = self.tables.datasets_of[i]
                counts[d] = counts.get(d, 0) + 1
            tasks.append(
                TaskEpisode(
                    name=name,
                    weight=float(weight),
                    support=support,
                    query=query,
                    support_datasets=counts,
                    query_sequence_disjoint=seq_disjoint,
                    query_overlaps_support=overlap,
                )
            )
        return EpisodeSpec(spec_id=int(spec_id), tasks=tasks)

    def plan_for_rank(self, spec: EpisodeSpec, rank: int) -> list[StreamItem]:
        """The clips rank ``rank`` consumes for ``spec``, in order: per task, its query clips first (so the
        zero-shot diagnostic can run at theta), then ``clips_per_rank`` clips for each of the ``inner_steps``."""
        if not 0 <= int(rank) < self.world_size:
            raise ValueError(f"rank {rank} outside world of size {self.world_size}")
        cpr, qpr, world = int(self.cfg.clips_per_rank), int(self.cfg.query_clips_per_rank), self.world_size
        items: list[StreamItem] = []
        for ti, task in enumerate(spec.tasks):
            for q in range(qpr):
                items.append(StreamItem(spec.spec_id, ti, ROLE_QUERY, -1, q, task.query[rank * qpr + q]))
            n_support = len(task.support)
            for s in range(int(self.cfg.inner_steps)):
                for c in range(cpr):
                    pos = (s * world * cpr + rank * cpr + c) % n_support
                    items.append(StreamItem(spec.spec_id, ti, ROLE_SUPPORT, s, c, task.support[pos]))
        return items

    def items_per_episode(self, spec: EpisodeSpec) -> int:
        return len(spec.tasks) * self.cfg.items_per_rank_per_task()


__all__ = [
    "DATASET_SCHEDULES",
    "EPISODE_MODES",
    "LOSS_MODES",
    "MIXED_NAME",
    "ROLE_QUERY",
    "ROLE_SUPPORT",
    "ClipTables",
    "EpisodeSpec",
    "ReptileConfig",
    "ReptileEpisodeSampler",
    "StreamItem",
    "TaskEpisode",
    "eps_at",
    "load_manifest_rows",
    "unique_rows",
]
