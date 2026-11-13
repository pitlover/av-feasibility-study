# STAGE1_MULTISOURCE_FPS_V1 (local patch; no remote writes)
# V2_META_COMPAT_PATCH_20261108_V21
"""Dataset discovery, stage-aware selection and preflight validation.

No Cosmos3 import is needed for planning. Original processed files are never modified.
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
from cosmos3prep.contracts import action_frame, action_pose_path

N_FRAMES, N_ACTIONS, ACTION_DIM = 61, 60, 9
DATASETS = ('kitti', 'waymo', 'av2', 'nuscenes')


def load_index(root: Path) -> list[dict]:
    path = Path(root) / 'index.jsonl'
    if not path.is_file():
        raise FileNotFoundError(f'Missing preprocessing index: {path}')
    rows = []
    for i, line in enumerate(path.read_text(encoding='utf8').splitlines(), start=1):
        if not line.strip():
            continue
        r = json.loads(line)
        ds = r.get('dataset')
        if ds not in DATASETS:
            raise ValueError(f'{path}:{i}: unsupported dataset {ds!r}')
        rows.append(r)
    if not rows:
        raise ValueError(f'Empty dataset index: {path}')
    ids = [r['clip_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate clip_id in preprocessing index; rebuild index first')
    return rows


def preflight(r: dict, *, max_dt_error_s: float = 0.07, check_video: bool = False) -> dict:
    """Reject missing, malformed, inconsistent or out-of-time clips (strict at runtime)."""
    paths = {k: Path(r[k]).expanduser() for k in ('video', 'action_file', 'meta')}
    for k, p in paths.items():
        if not p.is_file():
            raise FileNotFoundError(f'{r["clip_id"]}: missing {k}: {p}')
    meta = json.loads(paths['meta'].read_text(encoding='utf8'))
    for k in ('clip_id', 'dataset', 'split'):
        if meta[k] != r[k]:
            raise ValueError(f'{r["clip_id"]}: index/metadata mismatch {k}')
    if action_frame(meta) not in ('ego','camera'):
        raise ValueError(f'{r["clip_id"]}: unknown action_frame')
    if meta.get('pose_convention') != 'backward_framewise':
        raise ValueError(f'{r["clip_id"]}: expected backward_framewise, got {meta.get("pose_convention")}')
    if meta.get('action_normalization') != 'none':
        raise ValueError(f'{r["clip_id"]}: expected unnormalized 9D GT')
    nominal_fps = meta.get('fps')
    if type(nominal_fps) is not int or nominal_fps not in (10, 12, 20) or meta.get('frame_count') != 61 or meta.get('action_count') != 60:
        raise ValueError(f'{r["clip_id"]}: expected 10/12/20fps, 61 frames and 60 actions')
    a = np.load(paths['action_file'], mmap_mode='r', allow_pickle=False)
    if a.shape != (N_ACTIONS, ACTION_DIM) or not np.isfinite(a).all():
        raise ValueError(f'{r["clip_id"]}: malformed action {a.shape}')
    ts_path = paths['meta'].parent / 'timestamps_s.npy'
    if not ts_path.is_file():
        raise FileNotFoundError(ts_path)
    ts = np.load(ts_path, allow_pickle=False)
    if ts.shape != (N_FRAMES,) or not np.isfinite(ts).all() or not np.all(np.diff(ts) > 0):
        raise ValueError(f'{r["clip_id"]}: missing/nonmonotonic timestamps')
    dt = np.diff(ts)
    expected_dt = 1.0 / nominal_fps
    # Native 20Hz and nuScenes 12Hz need a tighter gap bound than nominal 10Hz.
    tolerance = min(max_dt_error_s, {10: 0.07, 12: 0.04, 20: 0.025}[nominal_fps])
    dt_error = float(np.max(np.abs(dt - expected_dt)))
    if dt_error > tolerance:
        raise ValueError(f'{r["clip_id"]}: max |dt-1/{nominal_fps}|={dt_error:.4f}s (tolerance={tolerance:.4f}s)')
    if check_video:
        import cv2
        cap = cv2.VideoCapture(str(paths['video']))
        if not cap.isOpened():
            raise ValueError(f'{r["clip_id"]}: MP4 cannot be decoded')
        cnt = 0
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            cnt += 1
        cap.release()
        if cnt != N_FRAMES or abs(video_fps - nominal_fps) > 0.01:
            raise ValueError(f'{r["clip_id"]}: MP4 {cnt} frames at {video_fps}fps (expected {nominal_fps})')
    # V2 camera-action GT uses poses_local_virtual_camera.npy.
    if action_frame(meta) == 'camera':
        action_pose_path(meta, paths['meta'].parent)
    return {'max_dt_error_s': dt_error, 'mean_dt_s': float(dt.mean()), 'fps': nominal_fps}


def select_stage(rows: list[dict], stage: str, selected: list[str], split: str = 'train') -> list[dict]:
    allowed = set(DATASETS) if stage == 'pretrain' else {'av2'}
    if not selected or len(set(selected)) != len(selected) or set(selected) - allowed:
        raise ValueError(f'{stage}: datasets must be a nonempty subset of {sorted(allowed)}')
    result = [r for r in rows if r['dataset'] in selected and r['split'] == split]
    if not result:
        raise ValueError(f'{stage}: no {split} clips for {selected}; available split counts: {dict(Counter((r["dataset"],r["split"]) for r in rows))}')
    return result


def mix_rows(rows: list[dict], weights: dict[str, float], count: int, seed: int) -> list[dict]:
    """Deterministic rank-independent mix, dataset proportions controlled by weights.
    Samples can repeat when requested epoch size exceeds available unique clips.
    """
    if count <= 0:
        raise ValueError('count must be positive')
    grouped = {name: [r for r in rows if r['dataset'] == name] for name in weights}
    if any(not v for v in grouped.values()):
        raise ValueError('No rows found for a requested dataset in the selected split')
    if any(not math.isfinite(w) or w <= 0 for w in weights.values()):
        raise ValueError('All dataset mixing weights must be positive and finite')
    total = sum(weights.values())
    quota = {k: count * w / total for k, w in weights.items()}
    picks = {k: int(math.floor(n)) for k, n in quota.items()}
    rest = count - sum(picks.values())
    for k in sorted(picks, key=lambda n: (quota[n] - picks[n], n), reverse=True)[:rest]:
        picks[k] += 1
    rng = random.Random(seed)
    out = []
    for k, n in picks.items():
        bucket = list(grouped[k])
        rng.shuffle(bucket)
        for i in range(n):
            if i % len(bucket) == 0 and i:
                rng.shuffle(bucket)
            out.append(bucket[i % len(bucket)])
    rng.shuffle(out)
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
