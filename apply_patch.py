#!/usr/bin/env python3
"""Apply V2 camera GT compatibility FIX LOCALLY without writing to GitHub.

Usage:
    python apply_patch.py --root /path/to/av-feasibility-sync-test --check
    python apply_patch.py --root /path/to/av-feasibility-sync-test --apply

Strict matching; refuses unknown source versions. Existing outputs/checkpoints
are never touched. Backs up changed .py files before updating atomically.
"""
from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import py_compile
import shutil
import tempfile
import sys

MARKER = '# V2_META_COMPAT_PATCH_20261108\n'


def replace_once(s: str, old: str, new: str, path: str) -> str:
    matches = s.count(old)
    if matches != 1:
        raise RuntimeError(f'{path}: expected 1 anchor, found {matches}: {old[:115]!r}')
    return s.replace(old, new, 1)


def patched(path: str, s: str) -> str:
    if s.startswith(MARKER):
        return s
    if path == 'cosmos3prep/cli.py':
        s = replace_once(s,
            'from .geometry import decode_local, validate_pose_array',
            'from .geometry import decode_local, validate_pose_array\nfrom .contracts import action_frame, action_pose_path', path)
        s = replace_once(s,
            "p = np.load(folder / ('poses_local_camera.npy' if meta.get('action_frame') == 'camera' else 'poses_local_ego.npy'))",
            'p = np.load(action_pose_path(meta, folder), allow_pickle=False)', path)
        s = replace_once(s,
            '        assert terr < 0.002, (mp, terr)',
            "        if terr >= 0.002:\n            raise ValueError(f'{mp}: {action_frame(meta)} GT roundtrip translation error={terr:.6f} m')", path)
    elif path == 'cosmos3av/data.py':
        s = replace_once(s, 'import numpy as np',
            'import numpy as np\nfrom cosmos3prep.contracts import action_frame, action_pose_path', path)
        s = replace_once(s, "if meta.get('action_frame', 'ego') not in ('ego','camera'):",
            "if action_frame(meta) not in ('ego','camera'):", path)
        s = replace_once(s,
            "    pose_name = 'poses_local_camera.npy' if meta.get('action_frame') == 'camera' else 'poses_local_ego.npy'\n    # Legacy synthetic manifests may only contain actions and timestamps.\n    # Camera-v4 contract strictly requires a camera-relative pose file.\n    if meta.get('action_frame') == 'camera' and not (paths['meta'].parent / pose_name).is_file():\n        raise FileNotFoundError(paths['meta'].parent / pose_name)",
            "    # V2 names its camera GT poses_local_virtual_camera.npy. Do not\n    # confuse the 9D camera actions with poses_local_ego.npy.\n    if action_frame(meta) == 'camera':\n        action_pose_path(meta, paths['meta'].parent)", path)
    elif path == 'cosmos3av/cli.py':
        s = replace_once(s, 'from .data import load_index, preflight, select_stage, mix_rows, write_jsonl',
            'from .data import load_index, preflight, select_stage, mix_rows, write_jsonl\nfrom cosmos3prep.contracts import action_frame', path)
        s = replace_once(s,
            "frames = {json.loads(Path(r['meta']).read_text(encoding='utf8')).get('action_frame', 'ego')\n              for r in train_rows}",
            "frames = {action_frame(json.loads(Path(r['meta']).read_text(encoding='utf8')))\n              for r in train_rows}", path)
    elif path == 'cosmos3av/framework_dataset.py':
        s = replace_once(s, 'import numpy as np',
            'import numpy as np\nfrom cosmos3prep.contracts import action_frame', path)
        s = replace_once(s,
            "if meta.get('action_frame') != 'camera' or meta.get('action_normalization') != 'none':",
            "if action_frame(meta) != 'camera' or meta.get('action_normalization') != 'none':", path)
    elif path == 'cosmos3virtual/normalization.py':
        s = replace_once(s, 'import numpy as np',
            'import numpy as np\nfrom cosmos3prep.contracts import action_frame', path)
        s = replace_once(s, 'if meta.get("action_frame") != "camera":',
            'if action_frame(meta) != "camera":', path)
    elif path == 'cosmos3eval/data.py':
        s = replace_once(s,
            'from .geometry import check_pose_array, check_actions, decode_actions, rotation_geodesic_deg',
            'from .geometry import check_pose_array, check_actions, decode_actions, rotation_geodesic_deg\nfrom cosmos3prep.contracts import V2, action_frame as resolve_action_frame, action_pose_path, normalized_metadata', path)
        s = replace_once(s,
            "if m.get('action_encoding')!='translation3 + rot6d_column_based':",
            "if m.get('format') != V2 and m.get('action_encoding')!='translation3 + rot6d_column_based':", path)
        s = replace_once(s,
            "    action_frame=m.get('action_frame','ego')\n    if action_frame not in ('ego','camera'):\n        raise ValueError(f'{cid}: unknown action_frame {action_frame}')\n    pose_file = 'poses_local_camera.npy' if action_frame=='camera' else 'poses_local_ego.npy'\n    gt=check_pose_array(np.load(meta.parent/pose_file,allow_pickle=False))",
            "    frame=resolve_action_frame(m)\n    gt=check_pose_array(np.load(action_pose_path(m, meta.parent),allow_pickle=False))", path)
        s = replace_once(s,
            '    return dict(gt_pose=gt,gt_actions=acts,metadata=m,timestamps=timestamps,',
            '    return dict(gt_pose=gt,gt_actions=acts,metadata=normalized_metadata(m),timestamps=timestamps,', path)
    elif path == 'cosmos3eval/camera_render.py':
        s = replace_once(s,
            'from cosmos3virtual.camera import project_camera_xyz',
            'from cosmos3virtual.camera import project_camera_xyz\nfrom cosmos3prep.contracts import video_intrinsics', path)
        s = replace_once(s,
            "K=np.asarray(meta['K_video'],np.float64)",
            'K=video_intrinsics(meta)', path)
    elif path == 'cosmos3eval/cli.py':
        s = replace_once(s,
            'from .data import prepare,read_jsonl',
            'from .data import prepare,read_jsonl\nfrom cosmos3prep.contracts import action_frame, image_mode', path)
        s = replace_once(s,
            "signatures.add((meta.get('action_frame','ego'),meta.get('image_mode','letterbox')))",
            'signatures.add((action_frame(meta),image_mode(meta)))', path)
        s = replace_once(s,
            "    if mode == 'camera':\n        from .camera_render import render_camera_selection",
            "    if a.layout and mode != 'camera':\n        raise ValueError('--layout requires a virtual-camera GT selection')\n    if mode == 'camera':\n        from .camera_render import render_camera_selection", path)
    else:
        raise ValueError(f'Unexpected patch path: {path}')
    return MARKER + s


FILES = [
    'cosmos3prep/cli.py', 'cosmos3av/data.py', 'cosmos3av/cli.py',
    'cosmos3av/framework_dataset.py', 'cosmos3virtual/normalization.py',
    'cosmos3eval/data.py', 'cosmos3eval/camera_render.py', 'cosmos3eval/cli.py',
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path.cwd(), help='Existing cloned code root')
    actions = p.add_mutually_exclusive_group(required=True)
    actions.add_argument('--check', action='store_true', help='Check patches without writing')
    actions.add_argument('--apply', action='store_true', help='Backup and apply patches locally')
    args = p.parse_args()
    root = args.root.expanduser().resolve()
    bundle = Path(__file__).resolve().parent
    if not (root / 'cosmos3av' / 'cli.py').is_file() or not (root / 'cosmos3prep' / 'cli.py').is_file():
        p.error(f'{root} is not a compatible Cosmos AV source checkout')
    edits = {}
    for relative in FILES:
        source = root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        original = source.read_text(encoding='utf-8')
        revised = patched(relative, original)
        compile(revised, str(source), 'exec')
        if original != revised:
            edits[relative] = revised
        print(f'{"CHANGE" if original != revised else "ALREADY PATCHED"} {relative}')
    module = (bundle / 'cosmos3prep' / 'contracts.py').read_text(encoding='utf-8')
    compile(module, str(root / 'cosmos3prep/contracts.py'), 'exec')
    target = root / 'cosmos3prep' / 'contracts.py'
    if not target.is_file() or target.read_text(encoding='utf-8') != module:
        edits['cosmos3prep/contracts.py'] = module
        print('CHANGE cosmos3prep/contracts.py')
    tests = (bundle / 'tests' / 'test_v2_metadata_compat.py').read_text(encoding='utf-8')
    compile(tests, str(root / 'tests/test_v2_metadata_compat.py'), 'exec')
    test_target = root / 'tests' / 'test_v2_metadata_compat.py'
    if not test_target.is_file() or test_target.read_text(encoding='utf-8') != tests:
        edits['tests/test_v2_metadata_compat.py'] = tests
        print('CHANGE tests/test_v2_metadata_compat.py')
    if args.check:
        print(f'\n[CHECK OK] {len(edits)} local files require update. --apply will create backups.')
        return
    if not edits:
        print('[NO CHANGES] Already applied')
        return
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = root / '.v2_metadata_backup' / stamp
    for relative in edits:
        source = root / relative
        if source.is_file():
            dest = backup / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
    for relative, revised in edits.items():
        dest = root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + '.new')
        tmp.write_text(revised, encoding='utf-8')
        tmp.replace(dest)
    print(f'\n[APPLIED] {len(edits)} code files updated locally')
    print(f'[BACKUP] {backup} (only files that previously existed)')
    print('[NEXT] python -m pytest -q tests/test_v2_metadata_compat.py')
    print('[NEXT] python -m cosmos3virtual.cli validate --out "$VIRTUAL" --max-clips 3')
    print('No GitHub modifications or uploads were performed.')


if __name__ == '__main__':
    main()
