#!/usr/bin/env python3
"""Strict Stage2 TOML parser: CLI overrides > TOML > env > defaults.

Stage2 ALWAYS consumes an existing Stage1 run and checkpoint; never starts Stage1.
Python 3.11+ required (tomllib).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
import tomllib

# dotted-key -> (environment name, type, default)
SCHEMA = {
    'job.run_tag': ('RUN_TAG', str, 's2_av2_baseline_v1'),
    'job.eval_tag': ('EVAL_TAG', str, 'val_seed42_v1'),
    'job.run_step': ('RUN_STEP', str, 'config'),
    'paths.run_root': ('RUNS_ROOT', str, 'runs_stage2'),
    'paths.exports_root': ('EXPORTS_ROOT', str, 'exports'),
    'paths.evaluation_root': ('EVALUATION_ROOT', str, 'evaluation'),
    'paths.base': ('BASE', str, str(Path.cwd())),
    'paths.processed': ('PROCESSED', str, 'processed_virtual_native_v2'),
    'paths.framework': ('FRAMEWORK', str, 'cosmos-framework'),
    'paths.stage1_run': ('STAGE1_RUN', str, 'runs_stage1_full/s1_raw_v2'),
    'paths.stage1_checkpoint': ('STAGE1_CKPT', str,
        'runs_stage1_full/s1_raw_v2/outputs/cosmos3_av_id/pretrain/s1_raw_v2/checkpoints/iter_000003000'),
    'paths.wan_vae': ('WAN_VAE', str, ''),  # fallback to stage1_run/run.json
    'data.epoch_clips': ('EPOCH_CLIPS', int, 12000),
    'data.seed': ('SEED', int, 42),
    'data.datasets': ('DATASETS_JSON', list, ['av2']),
    'data.sources.kitti': ('SOURCE_KITTI', str, ''),
    'data.sources.waymo': ('SOURCE_WAYMO', str, ''),
    'data.sources.av2': ('SOURCE_AV2', str, ''),
    'data.sources.nuscenes': ('SOURCE_NUSCENES', str, ''),
    'data.weights.kitti': ('WEIGHT_KITTI', float, 1.0),
    'data.weights.waymo': ('WEIGHT_WAYMO', float, 1.0),
    'data.weights.av2': ('WEIGHT_AV2', float, 1.0),
    'data.weights.nuscenes': ('WEIGHT_NUSCENES', float, 1.0),
    'train.gpus': ('GPUS', int, 4),
    'train.max_iters': ('MAX_ITERS', int, 3000),
    'train.save_every': ('SAVE_EVERY', int, 500),
    'optimizer.lr': ('LR', float, 1e-5),
    'wandb.enabled': ('WANDB_ENABLED', bool, False),
    'wandb.mode': ('WANDB_MODE', str, 'disabled'),
    'wandb.project': ('WANDB_PROJECT', str, 'cosmos3_av_id'),
    'export.step': ('STEP', int, 0),
    'eval.clips_per_dataset': ('EVAL_CLIPS', int, 20),
    'eval.infer_gpus': ('INFER_GPUS', int, 1),
    # 'full' runs fresh infer; other modes reuse existing prediction files.
    'eval.mode': ('EVAL_MODE', str, 'full'),
    'eval.eval_dir': ('EVAL_DIR_OVERRIDE', str, ''),
    'eval.pred_dir': ('PRED_DIR_OVERRIDE', str, ''),
    'eval.ground_height_m': ('GROUND_HEIGHT_M', float, 1.7),
    'eval.horizon_steps': ('HORIZON_STEPS', int, 60),
    'eval.fps': ('RENDER_FPS', float, 10.0),
    'eval.layout': ('RENDER_LAYOUT', str, 'both'),
    'eval.render_enabled': ('RENDER_ENABLED', bool, False),
    'eval.render_max_clips': ('RENDER_MAX_CLIPS', int, 1),
    'eval.render_clip_ids': ('RENDER_CLIP_IDS_JSON', list, []),
}


def flatten(d: dict, parent: str = '') -> dict:
    out = {}
    for k, v in d.items():
        key = f'{parent}.{k}' if parent else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def read_override(s: str):
    if '=' not in s:
        raise ValueError(f'Expected section.key=value, got {s!r}')
    key, val = s.split('=', 1)
    if key not in SCHEMA:
        raise ValueError(f'Unsupported CLI key: {key}')
    try:
        val = tomllib.loads('value = ' + val)['value']
    except tomllib.TOMLDecodeError:
        pass
    return key, val


def coerce(k, raw):
    typ = SCHEMA[k][1]
    if typ == list:
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{k} must be a TOML array of clip IDs') from exc
        if not isinstance(raw, list) or any(not isinstance(v, str) or not v.strip() for v in raw):
            raise ValueError(f'{k} must be an array of nonempty string clip IDs')
        if len(set(raw)) != len(raw):
            raise ValueError(f'{k} contains duplicate clip IDs')
        return raw
    if typ == bool:
        if type(raw) == bool:
            return raw
        if isinstance(raw, str) and raw.lower() in {'true', 'false'}:
            return raw.lower() == 'true'
        raise ValueError(f'{k} must be true/false')
    if typ == int:
        if type(raw) == int:
            return raw
        if isinstance(raw, str) and re.fullmatch(r'\d+', raw):
            return int(raw)
        raise ValueError(f'{k} must be an integer')
    if typ == float:
        if type(raw) == bool:
            raise ValueError(f'{k} must be a number')
        try:
            f = float(raw)
        except (ValueError, TypeError) as e:
            raise ValueError(f'{k} must be a number') from e
        if not math.isfinite(f):
            raise ValueError(f'{k} must be finite')
        return f
    if not isinstance(raw, str) or '\n' in raw or '\x00' in raw or '\r' in raw:
        raise ValueError(f'{k} must be a single-line string')
    return raw


def resolve(path: str | None, overrides: list[str]) -> dict:
    parsed = {}
    if path:
        with Path(path).expanduser().resolve().open('rb') as f:
            parsed = flatten(tomllib.load(f))
    unknown = sorted(set(parsed).difference(SCHEMA))
    if unknown:
        raise ValueError('Unsupported TOML keys: ' + ', '.join(unknown))
    cli = dict(read_override(x) for x in overrides)
    values = {}
    for key, (env, typ, default) in SCHEMA.items():
        raw = default
        raw = parsed.get(key, raw)
        raw = cli.get(key, raw)
        values[key] = coerce(key, raw)
    selected = values['data.datasets']
    allowed = {'kitti', 'waymo', 'av2', 'nuscenes'}
    if not selected or len(set(selected)) != len(selected) or set(selected) - allowed:
        raise ValueError(f'data.datasets must be a unique nonempty subset of {sorted(allowed)}')
    for ds in selected:
        if values[f'data.weights.{ds}'] <= 0:
            raise ValueError(f'data.weights.{ds} must be positive')
    enabled, mode = values['wandb.enabled'], values['wandb.mode']
    if mode not in {'online', 'offline', 'disabled'}:
        raise ValueError('wandb.mode must be disabled/online/offline')
    if not enabled and mode != 'disabled':
        raise ValueError('wandb.enabled=false conflicts with wandb.mode != disabled')
    if enabled and mode == 'disabled':
        raise ValueError('wandb.enabled=true conflicts with wandb.mode=disabled')
    for key in ('train.max_iters','train.save_every','train.gpus', 'data.epoch_clips',
                'eval.clips_per_dataset','eval.infer_gpus','eval.horizon_steps', 'eval.render_max_clips'):
        if values[key] < 1:
            raise ValueError(f'{key} must be positive')
    if values['train.max_iters'] % values['train.save_every']:
        raise ValueError('train.max_iters must be divisible by train.save_every')
    if not 0 <= values['export.step'] <= values['train.max_iters']:
        raise ValueError('export.step must be 0..train.max_iters')
    if values['export.step'] and values['export.step'] % values['train.save_every']:
        raise ValueError('export.step must be a saved iteration')
    if values['export.step'] == 0:
        values['export.step'] = values['train.max_iters']
    if values['optimizer.lr'] <= 0 or values['eval.fps'] < 0:
        raise ValueError('optimizer.lr must be positive and eval.fps must be >=0 (0=per-clip auto)')
    if values['eval.horizon_steps'] > 60:
        raise ValueError('eval.horizon_steps must be 1..60')
    if values['eval.layout'] not in {'both','bev','projected'}:
        raise ValueError('eval.layout must be both/bev/projected')
    if values['eval.mode'] not in {'full', 'score', 'render', 'both'}:
        raise ValueError('eval.mode must be one of full/score/render/both')
    if not values['wandb.project'].strip():
        raise ValueError('wandb.project must not be blank')
    if values['job.run_step'] not in {'config','inspect','plan','train','export','prepare','oracle','infer','score','render','evaluate','all'}:
        raise ValueError('job.run_step is invalid')
    for key in ('paths.run_root','paths.exports_root','paths.evaluation_root'):
        path = values[key]
        if not path.strip() or path.strip() in {'/', '.', '..'} or '..' in Path(path).parts:
            raise ValueError(f'Unsafe output root: {key}={path!r}')
    for key in ('job.run_tag','job.eval_tag'):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', values[key]):
            raise ValueError(f'{key} allows only letters/numbers/_/-')
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--toml')
    p.add_argument('--format', choices=('json','env0','source-json'), default='env0')
    p.add_argument('overrides', nargs='*')
    args = p.parse_args()
    try:
        data = resolve(args.toml, args.overrides)
    except (ValueError,OSError,tomllib.TOMLDecodeError) as e:
        print(f'[TOML ERROR] {e}', file=sys.stderr)
        return 2
    if args.format == 'json':
        print(json.dumps(data, indent=2, ensure_ascii=False))
    elif args.format == 'source-json':
        selected = data['data.datasets']
        spec = {
            'base': data['paths.base'],
            'processed': data['paths.processed'],
            'datasets': selected,
            'sources': {ds: data[f'data.sources.{ds}'] for ds in selected},
            'weights': {ds: data[f'data.weights.{ds}'] for ds in selected},
        }
        print(json.dumps(spec, ensure_ascii=False))
    else:
        for key, (env, _, _) in SCHEMA.items():
            v = data[key]
            if isinstance(v, bool):
                v = 'true' if v else 'false'
            elif isinstance(v, list):
                v = json.dumps(v, ensure_ascii=False)
            sys.stdout.buffer.write(env.encode() + b'\0' + str(v).encode() + b'\0')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
