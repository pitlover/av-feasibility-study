"""Helpers shared by the Reptile tests: import the code tree and build a local copy of the bundled sample index."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAMPLE_ROOT = ROOT / 'processed_virtual_native_v2'
# The bundled index.jsonl is the collaborator's full index; only a few clips per dataset/split are bundled,
# and their paths are absolute on that server. Rewrite the prefix and keep the clips that exist here.
REMOTE_PREFIX = '/mnt/ufs/users/jaeha-lee/share/jiiyoung/cosmos3_ji/processed_virtual_native_v2'


def sample_rows() -> list[dict]:
    index = SAMPLE_ROOT / 'index.jsonl'
    if not index.is_file():
        return []
    out = []
    for line in index.read_text(encoding='utf8').splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        for k in ('video', 'meta', 'action_file'):
            r[k] = r[k].replace(REMOTE_PREFIX, str(SAMPLE_ROOT))
        if all(Path(r[k]).is_file() for k in ('video', 'meta', 'action_file')):
            out.append(r)
    return out


def have_sample_clips(min_per_dataset: int = 2) -> bool:
    counts: dict[str, int] = {}
    for r in sample_rows():
        if r['split'] == 'train':
            counts[r['dataset']] = counts.get(r['dataset'], 0) + 1
    return counts.get('kitti', 0) >= min_per_dataset and counts.get('waymo', 0) >= min_per_dataset


def write_local_index(dest: Path, rows: list[dict] | None = None) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    rows = sample_rows() if rows is None else rows
    path = dest / 'index.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf8')
    return path


def synthetic_rows(n_kitti: int = 20, n_waymo: int = 30, seqs_kitti: int = 4, seqs_waymo: int = 10) -> list[dict]:
    """Manifest rows without files (sampler tests): clip i of a dataset belongs to sequence i % n_sequences."""
    rows = []
    for i in range(n_kitti):
        rows.append(dict(clip_id=f'kitti_{i:03d}', dataset='kitti', sequence=f'k{i % seqs_kitti}', split='train',
                         video='', meta='', action_file=''))
    for i in range(n_waymo):
        rows.append(dict(clip_id=f'waymo_{i:03d}', dataset='waymo', sequence=f'w{i % seqs_waymo}', split='train',
                         video='', meta='', action_file=''))
    return rows
