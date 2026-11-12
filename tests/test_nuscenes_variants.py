"""Synthetic CPU/ffmpeg smoke tests for nuScenes 12Hz preprocessor.

Run from the project root:
    python -m pytest -q tests/test_nuscenes_variants.py
No nuScenes download or GPU is required.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from cosmos3prep.nuscenes_adapter import VAL_SCENES, iter_nuscenes, make_transform, stream_json_array
from cosmos3prep.three_variants import target_fps, native_runs, run_prepare, validate_clips


def _json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _create_fake_nuscenes(root: Path, n_frames: int = 73):
    """Two scenes, each with CAM_FRONT 12Hz keyframes+sweeps and a dummy LIDAR row."""
    root.mkdir(parents=True, exist_ok=True)
    meta = root / "v1.0-trainval"
    meta.mkdir()
    scenes, samples, sample_data, poses = [], [], [], []
    sensor_token = "sensor-cam-front"
    calib_token = "calibration-front"
    K = [[115.0, 0, 80.0], [0, 114.0, 45.0], [0, 0, 1]]
    for scene_idx, scene_name in enumerate(("scene-0001", "scene-0003")):
        scene_token = f"scene-token-{scene_idx}"
        first_sample, last_sample = None, None
        for i in range(n_frames):
            keyframe = i % 6 == 0
            sample_token = f"sample-{scene_idx}-{i}" if keyframe else f"sample-{scene_idx}-{i//6*6}"
            timestamp = 1_530_000_000_000_000 + scene_idx * 40_000_000 + round(i * (1e6 / 12))
            token = f"sd-{scene_idx}-{i}"
            pose_token = f"pose-{scene_idx}-{i}"
            rel_image = ("samples" if keyframe else "sweeps") + f"/CAM_FRONT/{token}.jpg"
            path = root / rel_image
            path.parent.mkdir(parents=True, exist_ok=True)
            img = np.empty((90, 160, 3), np.uint8)
            img[:, :, :] = (i % 255, scene_idx * 30, 100)
            assert cv2.imwrite(str(path), img)
            sample_data.append({
                "token": token,
                "sample_token": sample_token,
                "ego_pose_token": pose_token,
                "calibrated_sensor_token": calib_token,
                "timestamp": timestamp,
                "filename": rel_image,
                "height": 90, "width": 160,
                "prev": f"sd-{scene_idx}-{i-1}" if i else "",
                "next": f"sd-{scene_idx}-{i+1}" if i < n_frames-1 else "",
                "is_key_frame": keyframe,
            })
            poses.append({
                "token": pose_token, "timestamp": timestamp,
                "translation": [i * 0.1, 0.0, 0.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],
            })
            if keyframe:
                samples.append({"token": sample_token, "scene_token": scene_token, "timestamp": timestamp})
                if first_sample is None:
                    first_sample = sample_token
                last_sample = sample_token
        scenes.append({"name": scene_name, "token": scene_token,
                       "first_sample_token": first_sample, "last_sample_token": last_sample})
    # Inject irrelevant non-CAM_FRONT sensor record to exercise the filter.
    sample_data.insert(0, {"token": "lidar-unrelated", "calibrated_sensor_token": "lidar-calibration"})
    _json(meta / "scene.json", scenes)
    _json(meta / "sample.json", samples)
    _json(meta / "sensor.json", [{"token": sensor_token, "channel": "CAM_FRONT", "modality": "camera"}])
    _json(meta / "calibrated_sensor.json", [{
        "token": calib_token, "sensor_token": sensor_token,
        "translation": [1.5, 0.0, 1.8],
        "rotation": [1.0, 0.0, 0.0, 0.0],
        "camera_intrinsic": K,
    }])
    _json(meta / "sample_data.json", sample_data)
    _json(meta / "ego_pose.json", poses)
    return np.asarray(K)


def test_official_split():
    assert len(VAL_SCENES) == 150
    assert "scene-0003" in VAL_SCENES
    assert "scene-0001" not in VAL_SCENES


def test_fps_modes():
    assert target_fps("nuscenes", "original_native") == 12
    assert target_fps("nuscenes", "virtual_native") == 12
    with pytest.raises(ValueError, match="two 12Hz variants"):
        target_fps("nuscenes", "original_10hz")


def test_camera_transform_quaternion():
    T = make_transform([1, 2, 3], [1, 0, 0, 0])
    np.testing.assert_allclose(T, np.array([[1,0,0,1],[0,1,0,2],[0,0,1,3],[0,0,0,1]]))


def test_metadata_sweeps_and_split(tmp_path):
    root = tmp_path / "nuscenes"
    K = _create_fake_nuscenes(root)
    assert sum(1 for _ in stream_json_array(root / "v1.0-trainval" / "sample_data.json")) == 147
    scenes = list(iter_nuscenes(root))
    assert [(a, b) for a,b,_ in scenes] == [("scene-0001", "train"), ("scene-0003", "val")]
    for _, _, frames in scenes:
        assert len(frames) == 73
        assert sum("/samples/" in str(f.image) for f in frames) == 13
        assert sum("/sweeps/" in str(f.image) for f in frames) == 60
        assert len(native_runs(frames, "nuscenes", 12)) == 1
        np.testing.assert_allclose(frames[0].K, K)
        np.testing.assert_allclose(frames[0].ego_camera[:3,3], [1.5,0,1.8])


@pytest.mark.parametrize("variant,expected_wh", [
    ("original_native", (160, 90)),
    ("virtual_native", (96, 64)),
])
def test_end_to_end_one_clip(tmp_path, variant, expected_wh):
    root = tmp_path / "nuScenes"
    K = _create_fake_nuscenes(root)
    out = tmp_path / "output"
    class Args:
        dataset = "nuscenes"
        camera = "CAM_FRONT"
        sequences = None
        poses_dir = None
        calib_root = None
        compression = ""
        max_logs = 1
        max_clips = 1
        stride = 60
        tolerance_ms = 35.0
        encoder = "local"
        virtual_width = 96
        virtual_height = 64
        virtual_hfov_deg = 45.0
        crf = 24
        preset = "ultrafast"
        overwrite = False
        strict = True
    a = Args()
    a.root = str(root)
    a.out_root = str(out)
    a.variant = variant
    assert run_prepare(a) == 1
    folder = next((out / variant / "nuscenes" / "train").glob("*"))
    meta = json.loads((folder / "metadata.json").read_text())
    assert meta["fps"] == 12
    assert meta["video_size_wh"] == list(expected_wh)
    assert meta["source_image_size_wh"] == [160, 90]
    assert meta["actual_capture_duration_s"] == pytest.approx(5, abs=.002)
    assert np.load(folder / "actions_raw_9d.npy").shape == (60,9)
    K_video = np.load(folder / "K_video_per_frame.npy")
    if variant == "original_native":
        np.testing.assert_allclose(K_video[0], K)
    else:
        assert K_video[0,0,2] == 48
        assert K_video[0,1,2] == 32
    assert validate_clips(out / variant, max_clips=0)["verified_clips"] == 1


def test_end_to_end_official_trainval(tmp_path):
    root = tmp_path / "nuScenes"
    _create_fake_nuscenes(root)
    out = tmp_path / "output"
    class Args:
        dataset = "nuscenes"
        camera = "CAM_FRONT"
        sequences = None
        poses_dir = None
        calib_root = None
        compression = ""
        max_logs = 0
        max_clips = 0
        stride = 60
        tolerance_ms = 35.0
        encoder = "local"
        virtual_width = 96
        virtual_height = 64
        virtual_hfov_deg = 45.0
        crf = 24
        preset = "ultrafast"
        overwrite = False
        strict = True
    a = Args()
    a.root = str(root)
    a.out_root = str(out)
    a.variant = "original_native"
    assert run_prepare(a) == 2
    idx = out / "original_native" / "index.jsonl"
    rows = [json.loads(line) for line in idx.read_text().splitlines()]
    assert len(rows) == 2
    assert {r["split"] for r in rows} == {"train", "val"}
    assert validate_clips(idx.parent, max_clips=0)["verified_clips"] == 2
