"""V2 legacy metadata regression tests.

Run in the project root after apply_patch.py. No live datasets are changed.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pytest

from cosmos3prep.contracts import (
    V2, action_frame, action_pose_path, image_mode, normalized_metadata,
    video_intrinsics,
)


def _make_v2_clip(tmp_path):
    clip = tmp_path / 'av2' / 'train' / 'av2_test_000000'
    clip.mkdir(parents=True)
    name = clip.name
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], 61, axis=0)
    poses[:, 2, 3] = np.arange(61) * .8  # camera Z forward, total 48 m
    ego = np.repeat(np.eye(4, dtype=np.float64)[None], 61, axis=0)
    ego[:, 0, 3] = np.arange(61) * .8  # ego X forward
    actions = np.zeros((60, 9), np.float32)
    actions[:, 2] = .8
    actions[:, 3] = 1.
    actions[:, 7] = 1.
    np.save(clip / 'poses_local_virtual_camera.npy', poses)
    np.save(clip / 'poses_local_camera.npy', poses)
    np.save(clip / 'poses_local_ego.npy', ego)
    np.save(clip / 'actions_raw_9d.npy', actions)
    np.save(clip / 'timestamps_s.npy', np.arange(61, dtype=np.float64) / 10.)
    (clip / 'video.mp4').write_bytes(b'stub only; metadata checks do not decode')
    meta = dict(format=V2, clip_id=name, dataset='av2', split='train', sequence='test',
                pose_convention='backward_framewise', action_encoder='local_cosmos_compatible',
                action_normalization='none', pose_local_file='poses_local_virtual_camera.npy',
                fps=10, frame_count=61, action_count=60, camera='stereo_front_left',
                K_virtual=[[772.548339959, 0, 320], [0, 772.548339959, 192], [0, 0, 1]])
    (clip / 'metadata.json').write_text(json.dumps(meta))
    row = dict(clip_id=name, dataset='av2', split='train', sequence='test',
               video=str(clip / 'video.mp4'), meta=str(clip / 'metadata.json'),
               action_file=str(clip / 'actions_raw_9d.npy'))
    return clip, meta, row


def test_v2_camera_frame_without_explicit_action_frame(tmp_path):
    clip, meta, _ = _make_v2_clip(tmp_path)
    assert 'action_frame' not in meta
    assert action_frame(meta) == 'camera'
    assert image_mode(meta) == 'virtual'
    assert action_pose_path(meta, clip).name == 'poses_local_virtual_camera.npy'
    assert normalized_metadata(meta)['action_frame'] == 'camera'
    assert video_intrinsics(meta)[0, 0] == pytest.approx(772.548339959)


def test_v2_preflight_and_score_gt_roundtrip(tmp_path):
    _, _, row = _make_v2_clip(tmp_path)
    from cosmos3av.data import preflight
    from cosmos3eval.data import verify_row
    assert preflight(row)['max_dt_error_s'] < 1e-8
    evaluation = verify_row(row)
    assert evaluation['metadata']['action_frame'] == 'camera'
    assert "action_encoding" not in json.loads(Path(row['meta']).read_text())
    assert evaluation['metadata']['image_mode'] == 'virtual'
    assert evaluation['max_gt_roundtrip_m'] < 1e-4
    assert evaluation['max_gt_roundtrip_deg'] < 1e-4
    assert evaluation['gt_pose'][-1, 2, 3] == pytest.approx(48.)


def test_v2_mismatched_ego_is_rejected(tmp_path):
    clip, meta, _ = _make_v2_clip(tmp_path)
    meta['action_frame'] = 'ego'
    with pytest.raises(ValueError, match='camera-frame'):
        action_frame(meta)
    meta.pop('action_frame')
    meta['pose_local_file'] = 'poses_local_ego.npy'
    with pytest.raises(ValueError, match='unrecognized'):
        action_pose_path(meta, clip)


def test_legacy_noncamera_stays_ego(tmp_path):
    clip = tmp_path / 'clip'
    clip.mkdir()
    (clip / 'poses_local_ego.npy').write_bytes(b'stub')
    meta = {'format': 'older_ego_format'}
    assert action_frame(meta) == 'ego'
    assert action_pose_path(meta, clip).name == 'poses_local_ego.npy'


def test_v2_explicit_unknown_encoding_is_rejected(tmp_path):
    clip, meta, row = _make_v2_clip(tmp_path)
    from cosmos3eval.data import verify_row
    meta['action_encoding'] = 'unsupported_other_action_format'
    (clip / 'metadata.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='unsupported V2 action encoding'):
        verify_row(row)


def test_v2_cli_validate_uses_camera_pose(tmp_path, monkeypatch):
    clip, meta, row = _make_v2_clip(tmp_path)
    from cosmos3prep import cli

    class StubCapture:
        def __init__(self, path):
            self.index = 0
        def isOpened(self):
            return True
        def get(self, key):
            return 10.0
        def read(self):
            if self.index < 61:
                self.index += 1
                return True, np.zeros((8, 8, 3), dtype=np.uint8)
            return False, None
        def release(self):
            pass

    monkeypatch.setattr(cli.cv2, 'VideoCapture', StubCapture)
    cli.validate_output(tmp_path, max_clips=1)
