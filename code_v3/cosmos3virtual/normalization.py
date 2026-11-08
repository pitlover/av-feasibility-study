"""Explicit and reversible GT action translation scaling, fitted on train only.

NO claim is made that these factors match the proprietary NVIDIA pretrained
Cosmos3-Edge action normalizer. 6D rotation is preserved, not normalized.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

SCHEMA = 'cosmos3_av_translation_maxabs_v1'


def fit_train_maxabs(index_rows, dataset_names, *, min_scale=1e-5):
    if not dataset_names:
        raise ValueError('At least one dataset required')
    maximum = np.zeros(3, dtype=np.float64)
    n = 0
    clipped_ids = []
    for r in index_rows:
        if r['dataset'] not in dataset_names or r['split'] != 'train':
            continue
        a = np.load(r['action_file'], mmap_mode='r',allow_pickle=False)
        if a.shape != (60,9) or not np.isfinite(a).all():
            raise ValueError(f'Bad actions {r["clip_id"]}: {a.shape}')
        maximum = np.maximum(maximum, np.abs(a[:,:3]).max(axis=0))
        clipped_ids.append(r['clip_id'])
        n+=1
    if not n:
        raise ValueError('No training clips in chosen datasets')
    scale=np.maximum(maximum, min_scale)
    return dict(schema=SCHEMA, axes='optical xyz: right, down, forward',
                mode='train_maxabs_xyz',
                xyz_scale_m=scale.tolist(),
                rotation6d='raw_column_based_unchanged',
                train_datasets=sorted(set(dataset_names)), train_unique_clips=n,
                source_split='train_only',
                fit_note='EXPERIMENTAL calibration: not verified as original Cosmos3-Edge checkpoint normalizer',
                source_clip_ids_sha256=__import__('hashlib').sha256('\n'.join(sorted(clipped_ids)).encode()).hexdigest())


def read_norm(path):
    if not path:return None
    p=Path(path)
    obj=json.loads(p.read_text())
    if obj.get('schema') != SCHEMA:
        raise ValueError(f'Unsupported normalizer schema: {obj.get("schema")}')
    scale=np.asarray(obj['xyz_scale_m'], dtype=np.float64)
    if scale.shape != (3,) or not np.all(np.isfinite(scale)) or np.any(scale<=0):
        raise ValueError('Invalid action xyz scales')
    return obj


def transform_action(action, norm):
    a=np.asarray(action, dtype=np.float64)
    if a.shape[-1] not in (9,64):
        raise ValueError(f'Action width must be 9 or padded 64, got {a.shape}')
    if norm is None:return a.copy()
    scale=np.asarray(norm['xyz_scale_m'],dtype=np.float64)
    result=a.copy();result[...,0:3]/=scale
    return result


def inverse_action(action, norm):
    a=np.asarray(action, dtype=np.float64)
    if a.shape[-1] not in (9,64):
        raise ValueError(f'Action width must be 9 or padded 64, got {a.shape}')
    if norm is None:return a.copy()
    scale=np.asarray(norm['xyz_scale_m'],dtype=np.float64)
    result=a.copy();result[...,0:3]*=scale
    return result
