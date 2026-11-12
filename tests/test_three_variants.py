"""CPU-only smoke / geometry tests. Run from the av-feasibility-study repo root.

    python -m pytest -q tests/test_three_variants.py

The H.264 test requires FFmpeg. It uses fully synthetic camera frames, no
KITTI/Waymo/AV2 data, and tests encoder metadata + 61f/60-action geometry.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from cosmos3prep.core import Frame
from cosmos3prep.three_variants import (
    _video_geometry,
    native_runs,
    select_runs,
    target_fps,
    validate_clips,
    write_variant_clip,
)


def make_frames(tmp_path: Path, n: int, fps: int, w: int = 65, h: int = 47) -> list[Frame]:
    path = tmp_path / "frame.png"
    yy, xx = np.mgrid[:h, :w]
    bgr = np.stack([(xx * 3 % 255).astype('uint8'),
                    (yy * 5 % 255).astype('uint8'),
                    np.full((h, w), 60, dtype='uint8')], axis=-1)
    assert cv2.imwrite(str(path), bgr)
    K = np.array([[100., 0., w / 2], [0., 110., h / 2], [0., 0., 1.]])
    world_ego = np.eye(4)
    ego_camera = np.eye(4)
    ego_camera[:3, 3] = [1.2, 0.1, 1.6]
    frames = []
    for i in range(n):
        p = world_ego.copy()
        p[0, 3] = i * 0.03
        frames.append(Frame(
            t_s=i / fps, image=path, world_ego=p, ego_camera=ego_camera,
            K=K.copy(), distortion=np.zeros(5), source_id=f'sequence/{i:06d}.png',
        ))
    return frames


def test_target_fps():
    assert [target_fps('av2', v) for v in ('original_10hz', 'original_native', 'virtual_native')] == [10, 20, 20]
    assert target_fps('waymo', 'original_native') == 10
    assert target_fps('kitti', 'virtual_native') == 10


def test_sampling_av2(tmp_path):
    frames = make_frames(tmp_path, 121, 20)
    assert len(select_runs(frames, 'av2', 'original_10hz')[0]) == 61
    assert len(select_runs(frames, 'av2', 'original_native')[0]) == 121
    assert len(select_runs(frames, 'av2', 'virtual_native')[0]) == 121
    # A missing 20 Hz camera frame must create a run boundary, not a duplicate.
    frames.pop(50)
    runs = native_runs(frames, 'av2', 20)
    assert len(runs) == 2
    assert len(runs[0]) == 50


def test_native_K_is_not_scaled_by_h264_even_padding(tmp_path):
    frames = make_frames(tmp_path, 61, 20, w=65, h=47)
    source_wh, encoded_wh, K = _video_geometry(frames[0], 'original_native', 640, 384, 45)
    assert source_wh == (65, 47)
    assert encoded_wh == (66, 48)  # pad right/bottom by one pixel
    np.testing.assert_array_equal(K, frames[0].K)
    _, virt_wh, K_virtual = _video_geometry(frames[0], 'virtual_native', 640, 384, 45)
    assert virt_wh == (640, 384)
    assert not np.array_equal(K, K_virtual)


@pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='FFmpeg not installed')
@pytest.mark.parametrize('variant, fps', [('original_native', 20), ('virtual_native', 20), ('original_10hz', 10)])
def test_h264_native_and_virtual_videos(tmp_path, variant, fps):
    frames = make_frames(tmp_path, 61, fps)
    result = write_variant_clip(
        frames, dataset='av2', sequence='demo', split='val', variant=variant,
        first_index=0, out_root=tmp_path / 'outputs', encoder='local',
        width=64, height=40, hfov=45, crf=30, preset='ultrafast',
    )
    assert result is not None
    folder = Path(result['meta']).parent
    meta = json.loads(Path(result['meta']).read_text(encoding='utf-8'))
    assert meta['fps'] == fps
    assert meta['frame_count'] == 61 and meta['action_count'] == 60
    assert meta['ablation_variant'] == variant
    assert np.load(folder / 'actions_raw_9d.npy').shape == (60, 9)
    assert np.load(folder / 'K_source_per_frame.npy').shape == (61, 3, 3)
    if variant == 'virtual_native':
        assert meta['video_size_wh'] == [64, 40]
        assert meta['image_mode'] == 'virtual'
    else:
        assert meta['video_size_wh'] == [66, 48]
        assert meta['image_mode'] == 'original'
        np.testing.assert_array_equal(
            np.load(folder / 'K_video_per_frame.npy')[0], frames[0].K)
    assert validate_clips(tmp_path / 'outputs' / variant, 0, verify_video=True)['verified_clips'] == 1


def test_variant_output_paths_are_separate(tmp_path):
    frames = make_frames(tmp_path, 61, 20)
    assert target_fps('av2', 'original_native') == 20
    assert target_fps('av2', 'original_10hz') == 10
    # Identical clip ID is safe because every preset gets its own root.
    assert ((tmp_path / 'original_native') / 'av2' / 'val') != ((tmp_path / 'original_10hz') / 'av2' / 'val')
