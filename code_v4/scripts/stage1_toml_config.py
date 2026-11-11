#!/usr/bin/env python3
"""Resolve a safe Stage1 TOML into NUL-delimited environment assignments.

Precedence: CLI dotted assignments > TOML > existing ENV > defaults.
Unknown TOML keys fail loudly. Requires Python 3.11+ (tomllib).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tomllib
from pathlib import Path

# dotted-key -> (environment variable, Python type, default, env variable used)
SCHEMA = {
    "job.run_tag": ("RUN_TAG", str, "s1_full_raw_v1", "RUN_TAG"),
    "job.eval_tag": ("EVAL_TAG", str, "val_seed42_v1", "EVAL_TAG"),
    # step executed by the no-argument entrypoints configs/scripts/run_stage1_*.sh (stage1_full_train_eval.sh takes
    # the step on the command line and ignores this key); "config" only prints the resolved settings.
    "job.run_step": ("RUN_STEP", str, "config", "RUN_STEP"),
    "paths.base": ("BASE", str, str(Path.cwd()), "BASE"),
    "paths.framework": ("FRAMEWORK", str, "", "FRAMEWORK"),
    "paths.processed": ("PROCESSED", str, "", "PROCESSED"),
    "paths.template_run": ("TEMPLATE_RUN", str, "", "TEMPLATE_RUN"),
    "paths.edge_dcp": ("EDGE_DCP", str, "", "EDGE_DCP"),
    "paths.wan_vae": ("WAN_VAE", str, "", "WAN_VAE"),
    # output roots used by configs/scripts/run_stage1_*.sh (relative = under paths.base); the defaults reproduce the
    # fixed layout of stage1_full_train_eval.sh: runs_stage1_full/<run_tag>, exports/, evaluation/.
    "paths.runs_root": ("RUNS_ROOT", str, "runs_stage1_full", "RUNS_ROOT"),
    "paths.exports_root": ("EXPORTS_ROOT", str, "exports", "EXPORTS_ROOT"),
    "paths.evaluation_root": ("EVALUATION_ROOT", str, "evaluation", "EVALUATION_ROOT"),
    "data.epoch_clips": ("EPOCH_CLIPS", int, 12000, "EPOCH_CLIPS"),
    "data.seed": ("SEED", int, 42, "SEED"),
    "data.weights.kitti": ("KITTI_WEIGHT", float, 1., "KITTI_WEIGHT"),
    "data.weights.waymo": ("WAYMO_WEIGHT", float, 1., "WAYMO_WEIGHT"),
    "train.gpus": ("GPUS", int, 4, "GPUS"),
    "train.max_iters": ("MAX_ITERS", int, 3000, "MAX_ITERS"),
    "train.save_every": ("SAVE_EVERY", int, 500, "SAVE_EVERY"),
    "optimizer.lr": ("LR", float, 1e-5, "LR"),
    "export.step": ("STEP", int, 0, "STEP"),  # 0 means use train.max_iters
    "eval.clips_per_dataset": ("EVAL_CLIPS", int, 20, "EVAL_CLIPS"),
    "eval.infer_gpus": ("INFER_GPUS", int, 1, "INFER_GPUS"),
    "eval.mode": ("EVAL_MODE", str, "full", "EVAL_MODE"),
    "eval.eval_dir": ("EVAL_DIR_OVERRIDE", str, "", "EVAL_DIR_OVERRIDE"),
    "eval.pred_dir": ("PRED_DIR_OVERRIDE", str, "", "PRED_DIR_OVERRIDE"),
    "eval.ground_height_m": ("GROUND_HEIGHT_M", float, 1.7, "GROUND_HEIGHT_M"),
    "eval.horizon_steps": ("HORIZON_STEPS", int, 60, "HORIZON_STEPS"),
    "eval.fps": ("RENDER_FPS", float, 10.0, "RENDER_FPS"),
    "eval.layout": ("RENDER_LAYOUT", str, "both", "RENDER_LAYOUT"),
    "eval.render_enabled": ("RENDER_ENABLED", bool, True, "RENDER_ENABLED"),
    "eval.render_max_clips": ("RENDER_MAX_CLIPS", int, 1, "RENDER_MAX_CLIPS"),
    "eval.render_clip_ids": ("RENDER_CLIP_IDS_JSON", list, [], "RENDER_CLIP_IDS_JSON"),
    "wandb.mode": ("WANDB_MODE", str, "disabled", "WANDB_MODE"),
    "wandb.project": ("WANDB_PROJECT", str, "cosmos3_av_id", "WANDB_PROJECT"),
    "wandb.enabled": ("WANDB_ENABLED", bool, None, "WANDB_ENABLED"),
    # [reptile]: Stage 1 Reptile meta-pretraining instead of joint SFT (cosmos3av/reptile.py documents every knob).
    # With enabled=true, train.max_iters counts META-iterations and the final DCP is iter_<max_iters>.
    "reptile.enabled": ("REPTILE_ENABLED", bool, False, "REPTILE_ENABLED"),
    "reptile.episode_mode": ("REPTILE_EPISODE_MODE", str, "per_dataset", "REPTILE_EPISODE_MODE"),
    "reptile.dataset_schedule": ("REPTILE_DATASET_SCHEDULE", str, "alternate", "REPTILE_DATASET_SCHEDULE"),
    "reptile.inner_steps": ("REPTILE_INNER_STEPS", int, 10, "REPTILE_INNER_STEPS"),
    "reptile.inner_warmup_steps": ("REPTILE_INNER_WARMUP_STEPS", int, 2, "REPTILE_INNER_WARMUP_STEPS"),
    "reptile.inner_grad_clip": ("REPTILE_INNER_GRAD_CLIP", float, 1.0, "REPTILE_INNER_GRAD_CLIP"),
    "reptile.inner_reset_optimizer": ("REPTILE_INNER_RESET_OPTIMIZER", bool, True, "REPTILE_INNER_RESET_OPTIMIZER"),
    "reptile.meta_lr": ("REPTILE_META_LR", float, 0.5, "REPTILE_META_LR"),
    "reptile.meta_lr_min_ratio": ("REPTILE_META_LR_MIN_RATIO", float, 0.1, "REPTILE_META_LR_MIN_RATIO"),
    "reptile.meta_lr_warmup_iters": ("REPTILE_META_LR_WARMUP_ITERS", int, 0, "REPTILE_META_LR_WARMUP_ITERS"),
    "reptile.clips_per_rank": ("REPTILE_CLIPS_PER_RANK", int, 1, "REPTILE_CLIPS_PER_RANK"),
    "reptile.support_clips": ("REPTILE_SUPPORT_CLIPS", int, 0, "REPTILE_SUPPORT_CLIPS"),
    "reptile.query_clips_per_rank": ("REPTILE_QUERY_CLIPS_PER_RANK", int, 1, "REPTILE_QUERY_CLIPS_PER_RANK"),
    "reptile.query_sequence_disjoint": ("REPTILE_QUERY_SEQUENCE_DISJOINT", bool, True, "REPTILE_QUERY_SEQUENCE_DISJOINT"),
    "reptile.eval_query_every": ("REPTILE_EVAL_QUERY_EVERY", int, 10, "REPTILE_EVAL_QUERY_EVERY"),
    "reptile.loss_mode": ("REPTILE_LOSS_MODE", str, "total", "REPTILE_LOSS_MODE"),
}

RUN_STEPS = ("config", "inspect", "plan", "train", "export", "prepare", "oracle", "infer", "score", "render", "evaluate", "all")
REPTILE_EPISODE_MODES = ("per_dataset", "all_datasets", "mixed")
REPTILE_DATASET_SCHEDULES = ("alternate", "random")
REPTILE_LOSS_MODES = ("total", "action")
# keys that must not change between `plan` and `train` (mirrors check_frozen_config in stage1_full_train_eval.sh)
REPTILE_FROZEN_KEYS = tuple(k for k in SCHEMA if k.startswith("reptile."))


def validate_reptile(values: dict[str, object]) -> None:
    """Cross-field checks of the [reptile] table (the same rules as cosmos3av.reptile.ReptileConfig, kept
    dependency-free so the shell can fail before any Python package is imported)."""
    if values["reptile.episode_mode"] not in REPTILE_EPISODE_MODES:
        raise ValueError(f"reptile.episode_mode must be one of {REPTILE_EPISODE_MODES}")
    if values["reptile.dataset_schedule"] not in REPTILE_DATASET_SCHEDULES:
        raise ValueError(f"reptile.dataset_schedule must be one of {REPTILE_DATASET_SCHEDULES}")
    if values["reptile.loss_mode"] not in REPTILE_LOSS_MODES:
        raise ValueError(f"reptile.loss_mode must be one of {REPTILE_LOSS_MODES}")
    for k in ("reptile.inner_steps", "reptile.clips_per_rank", "reptile.query_clips_per_rank"):
        if values[k] < 1:
            raise ValueError(f"{k} must be >= 1")
    for k in ("reptile.inner_warmup_steps", "reptile.meta_lr_warmup_iters", "reptile.support_clips",
              "reptile.eval_query_every", "reptile.inner_grad_clip"):
        if values[k] < 0:
            raise ValueError(f"{k} must be >= 0")
    if values["reptile.inner_warmup_steps"] > values["reptile.inner_steps"]:
        raise ValueError("reptile.inner_warmup_steps must be <= reptile.inner_steps")
    if not (0.0 < values["reptile.meta_lr"] <= 1.0):
        raise ValueError("reptile.meta_lr (Reptile eps) must be in (0, 1]")
    if not (0.0 < values["reptile.meta_lr_min_ratio"] <= 1.0):
        raise ValueError("reptile.meta_lr_min_ratio must be in (0, 1]")
    if values["reptile.query_clips_per_rank"] % values["reptile.clips_per_rank"]:
        raise ValueError("reptile.query_clips_per_rank must be a multiple of reptile.clips_per_rank (packed batches never mix roles)")
    min_support = values["train.gpus"] * values["reptile.clips_per_rank"]
    if values["reptile.support_clips"] and values["reptile.support_clips"] < min_support:
        raise ValueError(f"reptile.support_clips must be 0 (fresh clips every step) or >= train.gpus * clips_per_rank = {min_support}")
    if values["reptile.episode_mode"] == "per_dataset" and values["reptile.dataset_schedule"] == "alternate":
        if abs(values["data.weights.kitti"] - values["data.weights.waymo"]) > 1e-9:
            raise ValueError("reptile.dataset_schedule='alternate' visits both datasets equally; use equal data.weights or dataset_schedule='random'")


def reptile_json(values: dict[str, object]) -> dict[str, object]:
    """The [reptile] table as the JSON cosmos3av.cli plan --reptile-config consumes (ReptileConfig fields only;
    plan adds datasets / dataset_weights / manifest itself)."""
    if not values["reptile.enabled"]:
        raise ValueError("reptile.enabled is false: this TOML is a joint-training (SFT) Stage 1 plan")
    out = {k.split(".", 1)[1]: v for k, v in values.items() if k.startswith("reptile.") and k != "reptile.enabled"}
    out["seed"] = values["data.seed"]
    return out


def flatten(d: dict, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    for key, val in d.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(val, dict):
            out.update(flatten(val, name))
        else:
            out[name] = val
    return out


def parse_override(text: str) -> tuple[str, object]:
    if "=" not in text:
        raise ValueError(f"Expected a dotted key assignment, e.g. train.max_iters=500; got {text!r}")
    k, v = text.split("=", 1)
    if k not in SCHEMA:
        raise ValueError(f"Unknown override key {k!r}. Valid keys: {', '.join(SCHEMA)}")
    try:
        val = tomllib.loads("value = " + v)["value"]
    except tomllib.TOMLDecodeError:
        val = v
    return k, val


def coerce(key: str, val: object) -> object:
    typ = SCHEMA[key][1]
    if typ is list:
        if isinstance(val, str):
            try:
                val = json.loads(val)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{key} expects a TOML array of clip IDs") from exc
        if not isinstance(val, list) or any(not isinstance(x, str) or not x.strip() for x in val):
            raise ValueError(f"{key} expects an array of nonempty string clip IDs")
        if len(set(val)) != len(val):
            raise ValueError(f"{key} contains duplicate clip IDs")
        return val
    if typ is bool:
        if type(val) is not bool:
            if isinstance(val, str) and val.lower() in ("true", "false"):
                return val.lower() == "true"
            raise ValueError(f"{key} expects a TOML boolean true/false")
        return val
    if typ is int:
        if type(val) is int:
            return val
        if isinstance(val, str) and re.fullmatch(r"[0-9]+", val):
            return int(val)
        raise ValueError(f"{key} expects an integer, got {val!r}")
    if typ is float:
        if isinstance(val, bool):
            raise ValueError(f"{key} expects a number")
        try:
            n = float(val)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{key} expects a number, got {val!r}") from exc
        if not math.isfinite(n):
            raise ValueError(f"{key} requires a finite number")
        return n
    if not isinstance(val, str):
        raise ValueError(f"{key} expects a string, got {val!r}")
    if "\x00" in val or "\n" in val or "\r" in val:
        raise ValueError(f"{key} cannot contain control newlines")
    return val


def read_config(path: str | None, overrides: list[str]) -> dict[str, object]:
    parsed = {}
    if path:
        path_obj = Path(path).expanduser().resolve()
        if not path_obj.is_file():
            raise FileNotFoundError(path_obj)
        with path_obj.open("rb") as f:
            parsed = flatten(tomllib.load(f))
    unknown = sorted(set(parsed) - SCHEMA.keys())
    if unknown:
        raise ValueError(f"Unsupported TOML key(s): {', '.join(unknown)}")
    cli = dict(parse_override(v) for v in overrides)
    values: dict[str, object] = {}
    for k, (env_name, typ, default, fallback_env) in SCHEMA.items():
        raw = default
        if fallback_env in os.environ and os.environ[fallback_env] != "":
            raw = os.environ[fallback_env]
        if k in parsed:
            raw = parsed[k]
        if k in cli:
            raw = cli[k]
        values[k] = None if raw is None else coerce(k, raw)

    enabled = values["wandb.enabled"]
    mode = values["wandb.mode"]
    explicit_mode = "wandb.mode" in parsed or "wandb.mode" in cli
    if mode not in {"disabled", "online", "offline"}:
        raise ValueError("wandb.mode must be disabled, online, or offline")
    if enabled is False:
        if explicit_mode and mode != "disabled":
            raise ValueError("wandb.enabled=false conflicts with wandb.mode=online/offline")
        values["wandb.mode"] = "disabled"
    elif enabled is True and mode == "disabled":
        if explicit_mode:
            raise ValueError("wandb.enabled=true conflicts with wandb.mode=disabled")
        values["wandb.mode"] = "online"

    for k in ("train.gpus", "train.max_iters", "train.save_every", "data.epoch_clips",
              "eval.clips_per_dataset", "eval.infer_gpus", "eval.horizon_steps", "eval.render_max_clips"):
        if values[k] <= 0:
            raise ValueError(f"{k} must be positive")
    if values["train.max_iters"] % values["train.save_every"]:
        raise ValueError("train.max_iters must be divisible by train.save_every to save final checkpoint")
    if values["optimizer.lr"] <= 0:
        raise ValueError("optimizer.lr must be positive")
    for k in ("data.weights.kitti", "data.weights.waymo"):
        if values[k] <= 0:
            raise ValueError(f"{k} must be positive")
    if values["export.step"] < 0:
        raise ValueError("export.step cannot be negative")
    if values["export.step"] == 0:
        values["export.step"] = values["train.max_iters"]
    if values["eval.horizon_steps"] > 60:
        raise ValueError("eval.horizon_steps must be 1..60")
    if values["eval.mode"] not in {"score", "render", "both", "full"}:
        raise ValueError("eval.mode must be score/render/both/full")
    if values["eval.layout"] not in {"both", "bev", "projected"}:
        raise ValueError("eval.layout must be one of both/bev/projected")
    if values["eval.fps"] <= 0:
        raise ValueError("eval.fps must be positive")
    if not values["wandb.project"].strip():
        raise ValueError("wandb.project must not be blank")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", values["job.run_tag"]):
        raise ValueError("job.run_tag must use only letters, digits, _ and -")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", values["job.eval_tag"]):
        raise ValueError("job.eval_tag must use only letters, digits, _ and -")
    if values["job.run_step"] not in RUN_STEPS:
        raise ValueError(f"job.run_step must be one of {'/'.join(RUN_STEPS)}")
    for k in ("paths.runs_root", "paths.exports_root", "paths.evaluation_root"):
        if not values[k].strip() or values[k].strip() == "/":
            raise ValueError(f"{k} must be a non-empty directory (relative to paths.base or absolute)")
    if values["reptile.enabled"]:
        validate_reptile(values)
    return values


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--toml", default=None)
    ap.add_argument("--format", choices=["env0", "json", "reptile-json"], default="env0",
                    help="env0: NUL-delimited env assignments; json: every resolved key; "
                         "reptile-json: the [reptile] table for cosmos3av.cli plan --reptile-config")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    try:
        values = read_config(args.toml, args.overrides)
        if args.format == "reptile-json":
            payload = reptile_json(values)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as ex:
        print(f"[TOML ERROR] {ex}", file=sys.stderr)
        return 2
    if args.format == "json":
        json.dump(values, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
    elif args.format == "reptile-json":
        json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
    else:
        for dotted, (env_name, _, _, _) in SCHEMA.items():
            value = values[dotted]
            if value is None:
                value = ""
            elif isinstance(value, bool):
                value = "true" if value else "false"
            elif isinstance(value, list):
                value = json.dumps(value, ensure_ascii=False)
            sys.stdout.buffer.write(env_name.encode() + b"\0" + str(value).encode() + b"\0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
