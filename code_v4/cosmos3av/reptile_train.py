"""Reptile meta-pretraining of Cosmos3-Edge on KITTI + Waymo (Stage 1 alternative to joint training).

One meta-iteration (see ``cosmos3av/reptile.py`` for the sampling modes)::

    for task t in episode(j):                       # one dataset, or every dataset (all_datasets), or a mixture
        model <- theta
        k inner steps of the Stage 1 optimizer on fresh clips of t (same FusedAdam / lr / multipliers / clip)
        d_t = model - theta
    theta <- theta + eps(j) * weighted_mean_t(d_t);  model <- theta

Everything below the meta loop is the joint baseline's stack: the ``action_inverse_dynamics_av_edge``
experiment (registered by ``cosmos3av.entry_reptile``), ``OmniMoTModel.training_step``, FSDP with fp32
master weights, ``DistributedCheckpointer`` (DCP ``checkpoints/iter_XXXXXXXXX/{model,optim,scheduler,trainer}``
+ ``latest_checkpoint.txt`` -> re-running the same command auto-resumes) and the packing data loader. The run
directory therefore looks exactly like a joint Stage 1 run (``config.yaml``, DCP layout) and Stage 2 /
export consume it unchanged. Differences: ``model.config.ema.enabled`` must be False (theta is the smoothed
quantity; ``cosmos3av.cli plan`` writes ``[model.ema] enabled = false``), ``trainer.max_iter`` counts
META-iterations, and the loss logged per iteration is the inner-loop (adaptation) loss.

Outputs (``<run>/outputs/cosmos3_av_id/pretrain/<run_tag>/``): ``checkpoints/`` (DCP = theta), ``config.yaml``,
``reptile_config.json``, ``reptile_train_log.jsonl`` (per-iteration scalars, also sent to W&B when enabled).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from loguru import logger as logging

from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
from cosmos_framework.utils import distributed, misc
from cosmos_framework.utils.config import Config
from cosmos_framework.utils.context_managers import data_loader_init, distributed_init, model_init
from cosmos_framework.utils.lazy_config import LazyConfig, instantiate

from .reptile import (
    ROLE_QUERY,
    ROLE_SUPPORT,
    ClipTables,
    EpisodeSpec,
    ReptileConfig,
    ReptileEpisodeSampler,
    eps_at,
    load_manifest_rows,
    unique_rows,
)
from .reptile_data import META_KEYS, build_reptile_loader
from .reptile_meta import (
    ReptileMetaState,
    capture_base_lrs,
    clip_grads,
    mean_or_nan,
    reduce_mean,
    reset_optimizer_state,
    set_lr_scale,
)

LOSS_TERM_KEYS = ("flow_matching_loss_vision", "flow_matching_loss_action")

# the clip stream is infinite; the DataLoader's length bookkeeping warning is noise (the stock trainer filters it too)
warnings.filterwarnings(
    "ignore",
    message=r"Length of IterableDataset .* was reported to be .* samples have been fetched.*",
    category=UserWarning,
)


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def select_loss(output_batch: dict[str, Any], total_loss: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "total":
        return total_loss
    loss = output_batch.get("flow_matching_loss_action")
    if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
        raise RuntimeError("flow_matching_loss_action carries no gradient -- is action_gen enabled and the batch action-bearing?")
    return loss


def loss_terms(output_batch: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in LOSS_TERM_KEYS:
        v = output_batch.get(k)
        if isinstance(v, torch.Tensor) and v.numel() == 1:
            out[k] = float(v.detach())
    return out


def batch_meta(batch: dict[str, Any]) -> dict[str, list[int]]:
    """The ``_reptile_*`` ints of every packed sample (``list[Tensor(1,)]`` after the packing loader)."""
    out: dict[str, list[int]] = {}
    for k in META_KEYS:
        if k not in batch:
            raise KeyError(f"packed batch has no {k!r}: the loader is not the Reptile clip stream")
        vals = batch[k]
        if isinstance(vals, torch.Tensor):
            vals = [vals]
        out[k] = [int(torch.as_tensor(v).reshape(-1)[0]) for v in vals]
    return out


def check_batch(batch: dict[str, Any], *, spec_id: int, task: int, role: int, step: int, expected_samples: int) -> dict[str, list[int]]:
    """Assert the pulled batch is the one the schedule expects (every rank runs the same deterministic plan,
    so a mismatch means the data stream and the trainer desynchronized)."""
    meta = batch_meta(batch)
    n = len(meta["_reptile_spec_id"])
    if n != expected_samples:
        raise RuntimeError(f"expected a packed batch of {expected_samples} clip(s), got {n}")
    for k, expected in (("_reptile_spec_id", spec_id), ("_reptile_task", task), ("_reptile_role", role)):
        if any(v != expected for v in meta[k]):
            raise RuntimeError(f"Reptile stream desync: {k}={meta[k]} but the trainer expects {expected} (spec {spec_id}, task {task}, role {role}, step {step})")
    if role == ROLE_SUPPORT and any(v != step for v in meta["_reptile_step"]):
        raise RuntimeError(f"Reptile stream desync: inner step {meta['_reptile_step']} != expected {step} (spec {spec_id}, task {task})")
    return meta


def _warm_up_checkpoint_collectives(world_size: int) -> None:
    """Open the NCCL point-to-point connections ``dcp.save`` needs while the GPU is still empty (the first
    save otherwise sets them up lazily on a near-full card and can fail with an unhandled CUDA error)."""
    if world_size <= 1:
        return
    rank = dist.get_rank()
    gathered: list[Any] | None = [None] * world_size if rank == 0 else None
    dist.gather_object({"rank": rank}, gathered, dst=0)
    scattered: list[Any] = [None]
    dist.scatter_object_list(scattered, [{"rank": r} for r in range(world_size)] if rank == 0 else None, src=0)
    dist.barrier()


def spec_table(sampler: ReptileEpisodeSampler, first: int, count: int) -> list[dict[str, Any]]:
    """Summaries of a few episodes (dry-run / logging)."""
    return [sampler.spec(j).summary() for j in range(first, first + count)]


# --------------------------------------------------------------------------------------------
# main loop
# --------------------------------------------------------------------------------------------
@logging.catch(reraise=True)
def launch(config: Config, cfg: ReptileConfig, args: argparse.Namespace) -> None:
    with distributed_init():
        distributed.init()
    config.validate()
    config.freeze()

    rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    is_rank0 = rank == 0
    job_dir = Path(config.job.path_local)
    if is_rank0:
        job_dir.mkdir(parents=True, exist_ok=True)
        try:
            LazyConfig.save_pkl(config, str(job_dir / "config.pkl"))
        except Exception as e:  # noqa: BLE001 - the yaml is what export reads
            logging.warning(f"config.pkl not written ({e})")
        LazyConfig.save_yaml(config, str(job_dir / "config.yaml"))
        cfg.save(job_dir / "reptile_config.json")
        logging.info(f"Job dir: {job_dir}")
        logging.info(f"Reptile config: {json.dumps(cfg.to_dict())}")
    dist.barrier()
    _warm_up_checkpoint_collectives(world_size)
    misc.set_random_seed(seed=config.trainer.seed, by_rank=True)
    torch.backends.cudnn.deterministic = config.trainer.cudnn.deterministic
    torch.backends.cudnn.benchmark = config.trainer.cudnn.benchmark

    # ---- model + the Stage 1 optimizer (FSDP shards, fp32 master weights) -------------------------
    with model_init():
        model = instantiate(config.model)
    model = model.to("cuda", memory_format=config.trainer.memory_format)
    model.on_train_start(config.trainer.memory_format)
    if model.config.ema.enabled:
        raise ValueError(
            "Reptile needs model.config.ema.enabled=False (theta replaces the EMA copy); cosmos3av.cli plan writes "
            "[model.ema] enabled = false into sft_config.toml -- recreate the plan with --reptile-config"
        )
    optimizer, scheduler = model.init_optimizer_scheduler(config.optimizer, config.scheduler)
    grad_scaler = torch.amp.GradScaler("cuda", **config.trainer.grad_scaler_args)
    checkpointer = instantiate(config.checkpoint.type, config.checkpoint, config.job, callbacks=None)
    # warm start from BASE_CHECKPOINT_PATH (checkpoint.load_path) or auto-resume from this job's latest DCP
    start_iter = int(checkpointer.load(model, optimizer, scheduler, grad_scaler))
    model.train()
    max_iter = int(config.trainer.max_iter)
    if start_iter >= max_iter:
        logging.warning(f"start_iter {start_iter} >= trainer.max_iter {max_iter}: nothing to do")
        return

    state = ReptileMetaState(model.net)
    trainable_params = list(state.params.values())
    base_lrs = capture_base_lrs(optimizer)
    if cfg.inner_reset_optimizer:
        reset_optimizer_state(optimizer)
    logging.info(
        f"theta: {state.numel:,} params in {len(state.params)} tensors, by group {state.numel_by_group()}; "
        f"inner base lrs per group: {[[f'{lr:.2e}' for lr in b] for b in base_lrs]}; start_iter={start_iter}"
    )

    # ---- data: rank-synchronized clip stream through the baseline's loader stack --------------------
    tables = ClipTables(unique_rows(load_manifest_rows(cfg.manifest)))
    sampler = ReptileEpisodeSampler(tables, cfg, world_size)
    logging.info(
        f"clip tables: {tables.summary()}; episode_mode={cfg.episode_mode}, support {sampler.support_size} clips/task "
        f"({cfg.inner_steps} steps x {world_size} ranks x {cfg.clips_per_rank}), query {sampler.query_size} clips/task"
    )
    with data_loader_init():
        loader = build_reptile_loader(config, args.reptile_config, world_size, start_iter)
    loader_iter = iter(loader)
    n_query_batches = int(cfg.query_clips_per_rank) // int(cfg.clips_per_rank)

    # ---- logging -----------------------------------------------------------------------------------
    log_every = max(1, int(config.trainer.logging_iter))
    save_every = max(1, int(config.checkpoint.save_iter))
    wandb_run = None
    if is_rank0 and config.job.wandb_mode != "disabled":
        try:
            import wandb

            id_file = job_dir / "wandb_id.txt"
            run_id = id_file.read_text().strip() if id_file.exists() else wandb.util.generate_id()
            id_file.write_text(run_id)
            wandb_run = wandb.init(
                project=config.job.project,
                group=config.job.group,
                name=config.job.name,
                id=run_id,
                resume="allow",
                dir=os.environ.get("WANDB_DIR") or str(job_dir),
                mode=config.job.wandb_mode,
                config={"reptile": cfg.to_dict(), "sft_toml": args.sft_toml, "max_iter": max_iter},
            )
        except Exception as e:  # noqa: BLE001
            logging.warning(f"wandb init failed ({e}); continuing without wandb")
    jsonl = open(job_dir / "reptile_train_log.jsonl", "a") if is_rank0 else None

    def _save(iteration: int) -> None:
        # model == theta here (meta_step restored it). Drop the inner Adam moments: a resume / Stage 2 warm start
        # only needs theta, and the allocator's cached blocks go back to CUDA for the save collectives.
        if cfg.inner_reset_optimizer:
            reset_optimizer_state(optimizer)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        checkpointer.save(model, optimizer, scheduler, grad_scaler, iteration=iteration)
        if is_rank0:
            logging.info(f"Saved theta at meta-iteration {iteration}: {job_dir / 'checkpoints' / f'iter_{iteration:09d}'}")

    def _fetch(spec_id: int, task: int, role: int, step: int) -> dict[str, Any]:
        batch = next(loader_iter)
        check_batch(batch, spec_id=spec_id, task=task, role=role, step=step, expected_samples=int(cfg.clips_per_rank))
        return misc.to(batch, device="cuda")

    def _eval_query(query_batches: list[dict[str, Any]], iteration: int) -> tuple[float, dict[str, float]]:
        """Rank-averaged loss of the held-out query clips (+ the flow-matching terms). Collective."""
        vals: list[float] = []
        terms: dict[str, list[float]] = {k: [] for k in LOSS_TERM_KEYS}
        with torch.no_grad():
            for batch in query_batches:
                out, total = model.training_step(batch, iteration)
                vals.append(float(select_loss(out, total, cfg.loss_mode).detach()))
                for k, v in loss_terms(out).items():
                    terms[k].append(v)
        return reduce_mean(vals), {k: reduce_mean(terms[k]) for k in LOSS_TERM_KEYS}  # fixed key order on every rank

    logging.info(f"Starting Reptile meta-training: iterations {start_iter} -> {max_iter}, world_size={world_size}")
    iteration = start_iter
    ema_adapted: float | None = None
    while iteration < max_iter:
        t_iter = time.time()
        spec: EpisodeSpec = sampler.spec(iteration)
        do_eval = cfg.eval_query_every > 0 and (iteration % cfg.eval_query_every == 0)
        t_data = t_inner = t_eval = 0.0
        task_records: list[dict[str, Any]] = []
        for ti, task in enumerate(spec.tasks):
            if ti > 0:
                state.restore_to_model()  # every task adapts from theta
            # the task's held-out clips come first in the stream: zero-shot loss at theta, adapted loss after
            t0 = time.time()
            query_batches = [_fetch(spec.spec_id, ti, ROLE_QUERY, -1) for _ in range(n_query_batches)]
            t_data += time.time() - t0
            t0 = time.time()
            q_zero, q_zero_terms = _eval_query(query_batches, iteration) if do_eval else (float("nan"), {})
            t_eval += time.time() - t0

            # ---- inner loop: k Stage 1 optimizer steps from theta on fresh clips of the task ----------
            if cfg.inner_reset_optimizer:
                reset_optimizer_state(optimizer)
            inner_losses: list[float] = []
            inner_gnorms: list[float] = []
            last_terms: dict[str, float] = {}
            for s in range(int(cfg.inner_steps)):
                t0 = time.time()
                batch = _fetch(spec.spec_id, ti, ROLE_SUPPORT, s)
                t_data += time.time() - t0
                t0 = time.time()
                scale = min(1.0, (s + 1) / cfg.inner_warmup_steps) if cfg.inner_warmup_steps > 0 else 1.0
                set_lr_scale(optimizer, base_lrs, scale)
                out, total = model.training_step(batch, iteration)
                loss = select_loss(out, total, cfg.loss_mode)
                backward_loss = out.get("_backward_loss", total) if cfg.loss_mode == "total" else loss
                grad_scaler.scale(backward_loss).backward()
                model.on_after_backward()
                inner_gnorms.append(clip_grads(trainable_params, cfg.inner_grad_clip, force_finite=True))
                model.on_before_optimizer_step(optimizer, scheduler, iteration=iteration)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                model.on_before_zero_grad(optimizer, scheduler, iteration=iteration)
                optimizer.zero_grad(set_to_none=True)
                inner_losses.append(float(loss.detach()))
                last_terms = loss_terms(out)
                t_inner += time.time() - t0

            t0 = time.time()
            q_adapted, q_adapted_terms = _eval_query(query_batches, iteration) if do_eval else (float("nan"), {})
            t_eval += time.time() - t0
            del query_batches
            disp = state.displacement_norms_by_group()
            if len(spec.tasks) > 1:
                state.accumulate_displacement(task.weight)
            rec: dict[str, Any] = {
                "name": task.name,
                "weight": task.weight,
                "support_loss_first": reduce_mean([inner_losses[0]]),
                "support_loss_last": reduce_mean([inner_losses[-1]]),
                "inner_grad_norm": mean_or_nan(inner_gnorms),
                "query_loss_zero_shot": q_zero,
                "query_loss": q_adapted,
                "adapt_delta": math.sqrt(sum(v * v for v in disp.values())),
                "adapt_delta_by_group": disp,
                "query_sequence_disjoint": task.query_sequence_disjoint,
            }
            for k in LOSS_TERM_KEYS:  # fixed key order: reduce_mean is a collective
                rec[f"support_{k}_last"] = reduce_mean([last_terms[k]] if k in last_terms else [])
            if do_eval:
                for k in LOSS_TERM_KEYS:
                    rec[f"query_{k}_zero_shot"] = q_zero_terms.get(k, float("nan"))
                    rec[f"query_{k}"] = q_adapted_terms.get(k, float("nan"))
            task_records.append(rec)

        # ---- Reptile meta step --------------------------------------------------------------------
        t0 = time.time()
        eps = eps_at(cfg, iteration, max_iter)
        meta_step_norm = state.meta_step(eps)  # theta <- theta + eps * d ; model <- theta
        t_meta = time.time() - t0
        iteration += 1
        log_now = iteration % log_every == 0 or iteration == start_iter + 1
        theta_norms = state.theta_norms_by_group() if log_now else None  # collective

        # ---- logging (values are rank-averaged; every rank saw the same tasks) --------------------
        if is_rank0:
            q_ad = mean_or_nan([r["query_loss"] for r in task_records])
            if not math.isnan(q_ad):
                ema_adapted = q_ad if ema_adapted is None else 0.98 * ema_adapted + 0.02 * q_ad
            record: dict[str, Any] = {
                "iteration": iteration,
                "eps": eps,
                "tasks": [r["name"] for r in task_records],
                "support_loss_first": mean_or_nan([r["support_loss_first"] for r in task_records]),
                "support_loss_last": mean_or_nan([r["support_loss_last"] for r in task_records]),
                "inner_grad_norm": mean_or_nan([r["inner_grad_norm"] for r in task_records]),
                "query_loss": q_ad,
                "query_loss_ema": ema_adapted,
                "query_loss_zero_shot": mean_or_nan([r["query_loss_zero_shot"] for r in task_records]),
                "adapt_delta": mean_or_nan([r["adapt_delta"] for r in task_records]),
                "meta_step_norm": meta_step_norm,
                "iter_time": time.time() - t_iter,
                "t_data": t_data,
                "t_inner": t_inner,
                "t_eval": t_eval,
                "t_meta": t_meta,
                "mem_peak_alloc_gb": (torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else 0.0,
                "mem_peak_reserved_gb": (torch.cuda.max_memory_reserved() / 2**30) if torch.cuda.is_available() else 0.0,
            }
            if not (math.isnan(record["query_loss"]) or math.isnan(record["query_loss_zero_shot"])):
                record["query_loss_gain"] = record["query_loss_zero_shot"] - record["query_loss"]
            for k in LOSS_TERM_KEYS:
                record[f"support_{k}_last"] = mean_or_nan([r[f"support_{k}_last"] for r in task_records])
                if do_eval:
                    record[f"query_{k}_zero_shot"] = mean_or_nan([r[f"query_{k}_zero_shot"] for r in task_records])
                    record[f"query_{k}"] = mean_or_nan([r[f"query_{k}"] for r in task_records])
            for grp in sorted({g for r in task_records for g in r["adapt_delta_by_group"]}):
                record[f"adapt_delta/{grp}"] = mean_or_nan([r["adapt_delta_by_group"].get(grp, float("nan")) for r in task_records])
            for r in task_records:
                record[f"support_loss_last/{r['name']}"] = r["support_loss_last"]
                record[f"adapt_delta/{r['name']}"] = r["adapt_delta"]
                if not math.isnan(r["query_loss"]):
                    record[f"query_loss/{r['name']}"] = r["query_loss"]
                    record[f"query_loss_zero_shot/{r['name']}"] = r["query_loss_zero_shot"]
            if theta_norms is not None:
                record.update({f"theta_norm/{g}": v for g, v in theta_norms.items()})
            if log_now:
                q_str = f"query {record['query_loss']:.4f} (zero-shot {record['query_loss_zero_shot']:.4f}) | " if do_eval else ""
                adapt_str = ", ".join(f"{k.split('/')[1]} {v:.3f}" for k, v in record.items() if k.startswith("adapt_delta/"))
                logging.info(
                    f"iter {iteration}/{max_iter} | {record['tasks']} | {q_str}"
                    f"support {record['support_loss_first']:.4f} -> {record['support_loss_last']:.4f} | "
                    f"inner-grad {record['inner_grad_norm']:.3f} | adapt {record['adapt_delta']:.3f} ({adapt_str}) | "
                    f"eps {eps:.3f} step {meta_step_norm:.3f} | {record['iter_time']:.1f}s (data {t_data:.1f} inner {t_inner:.1f} "
                    f"eval {t_eval:.1f} meta {t_meta:.1f}) | peak mem {record['mem_peak_alloc_gb']:.1f} GiB alloc / "
                    f"{record['mem_peak_reserved_gb']:.1f} GiB reserved"
                )
            if jsonl is not None:
                jsonl.write(json.dumps(record) + "\n")
                jsonl.flush()
            if wandb_run is not None:
                wandb_run.log(
                    {k: v for k, v in record.items() if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))},
                    step=iteration,
                )
        if iteration % save_every == 0 or iteration == max_iter:
            _save(iteration)

    if jsonl is not None:
        jsonl.close()
    if wandb_run is not None:
        wandb_run.finish()
    finalize = getattr(checkpointer, "finalize", None)
    if callable(finalize):
        finalize()
    logging.success(f"Reptile meta-training finished at meta-iteration {iteration}. theta (DCP): {job_dir / 'checkpoints'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reptile meta-pretraining of Cosmos3-Edge on KITTI + Waymo (Stage 1)")
    parser.add_argument("--sft-toml", required=True, help="Run TOML written by cosmos3av.cli plan (sft_config.toml).")
    parser.add_argument("--reptile-config", required=True, help="reptile_config.json written by cosmos3av.cli plan --reptile-config.")
    parser.add_argument("--dryrun", action="store_true", help="Print the resolved config and the first episodes; no GPU.")
    parser.add_argument("--dryrun-world-size", type=int, default=None, help="World size assumed by --dryrun (default: FSDP shard degree).")
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=[], help="Extra Hydra dotted-path overrides (after --).")
    args = parser.parse_args()

    config = load_experiment_from_toml(args.sft_toml, extra_overrides=args.opts)
    cfg = ReptileConfig.load(args.reptile_config)
    args.config = args.sft_toml

    if args.dryrun:
        world = args.dryrun_world_size or int(config.model.config.parallelism.data_parallel_shard_degree)
        tables = ClipTables(unique_rows(load_manifest_rows(cfg.manifest)))
        sampler = ReptileEpisodeSampler(tables, cfg, world)
        logging.info("Config:\n" + config.pretty_print(use_color=True))
        logging.info("Reptile config: " + json.dumps(cfg.to_dict(), indent=2))
        logging.info(f"clip tables: {tables.summary()}")
        logging.info(
            f"world {world}: support {sampler.support_size} clips/task, query {sampler.query_size} clips/task, "
            f"{sampler.cfg.items_per_rank_per_task()} clips per rank per task; eps {eps_at(cfg, 0, int(config.trainer.max_iter)):.3f} -> "
            f"{eps_at(cfg, int(config.trainer.max_iter), int(config.trainer.max_iter)):.3f} over {int(config.trainer.max_iter)} meta-iterations"
        )
        logging.info("First episodes: " + json.dumps(spec_table(sampler, 0, 3), indent=2))
        return
    launch(config, cfg, args)
    if os.environ.get("COSMOS_EXIT_WITHOUT_FINALIZE", "").lower() in ("1", "true"):
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
