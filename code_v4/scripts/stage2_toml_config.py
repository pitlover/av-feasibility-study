#!/usr/bin/env python3
"""Resolve a safe Stage2 TOML into NUL-delimited environment assignments.

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

SCHEMA = {
    "job.run_tag": ("RUN_TAG", str, "repro_stage2_v1", "RUN_TAG"),
    "job.eval_tag": ("EVAL_TAG", str, "", "EVAL_TAG"),
    "paths.base": ("BASE", str, str(Path.cwd()), "BASE"),
    "paths.framework": ("FRAMEWORK", str, "", "FRAMEWORK"),
    "paths.processed": ("PROCESSED", str, "", "PROCESSED"),
    "paths.template_run": ("TEMPLATE_RUN", str, "", "TEMPLATE_RUN"),
    "paths.edge_dcp": ("EDGE_DCP", str, "", "EDGE_DCP"),
    "paths.wan_vae": ("WAN_VAE", str, "", "WAN_VAE"),
    "data.epoch_clips": ("EPOCH_CLIPS", int, 500, "EPOCH_CLIPS"),
    "data.seed": ("SEED", int, 42, "SEED"),
    "data.match_clip_id": ("MATCH_CLIP_ID", str, "av2_02678d04-cc9f-3148-9f95-1ba66347dff9_000000", "MATCH_CLIP_ID"),
    "data.weights.kitti": ("KITTI_WEIGHT", float, 1.0, "KITTI_WEIGHT"),
    "data.weights.waymo": ("WAYMO_WEIGHT", float, 1.0, "WAYMO_WEIGHT"),
    "train.gpus": ("GPUS", int, 4, "GPUS"),
    "train.max_iters": ("MAX_ITERS", int, 300, "MAX_ITERS"),
    "train.save_every": ("SAVE_EVERY", int, 50, "SAVE_EVERY"),
    "eval.clips_per_dataset": ("EVAL_CLIPS", int, 5, "EVAL_CLIPS"),
    "eval.infer_gpus": ("INFER_GPUS", int, 1, "INFER_GPUS"),
    "wandb.mode": ("WANDB_MODE", str, "disabled", "WANDB_MODE"),
    "wandb.enabled": ("WANDB_ENABLED", bool, None, "WANDB_ENABLED"),
    "wandb.project": ("WANDB_PROJECT", str, "cosmos3_av_id", "WANDB_PROJECT"),
}


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
              "eval.clips_per_dataset", "eval.infer_gpus"):
        if values[k] <= 0:
            raise ValueError(f"{k} must be positive")
    if values["train.max_iters"] % values["train.save_every"]:
        raise ValueError("train.max_iters must be divisible by train.save_every to save final checkpoint")
    for k in ("data.weights.kitti", "data.weights.waymo"):
        if values[k] <= 0:
            raise ValueError(f"{k} must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", values["job.run_tag"]):
        raise ValueError("job.run_tag must use only letters, digits, _ and -")
    eval_tag = str(values["job.eval_tag"])
    if eval_tag and not re.fullmatch(r"[A-Za-z0-9_-]+", eval_tag):
        raise ValueError("job.eval_tag must use only letters, digits, _ and -")
    if not eval_tag:
        values["job.eval_tag"] = values["job.run_tag"]
    return values


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--toml", default=None)
    ap.add_argument("--format", choices=["env0", "json"], default="env0")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    try:
        values = read_config(args.toml, args.overrides)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as ex:
        print(f"[TOML ERROR] {ex}", file=sys.stderr)
        return 2
    if args.format == "json":
        json.dump(values, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
    else:
        for dotted, (env_name, _, _, _) in SCHEMA.items():
            value = values[dotted]
            if value is None:
                value = ""
            elif isinstance(value, bool):
                value = "true" if value else "false"
            sys.stdout.buffer.write(env_name.encode() + b"\0" + str(value).encode() + b"\0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
