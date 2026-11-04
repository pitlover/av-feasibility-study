"""Small self-contained tests. No actual AV2 download, pandas/pyarrow installation, or GPU needed."""
from argparse import Namespace
from pathlib import Path
import json
import sys

import cv2
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import av2_loader
from camera import K_VIRTUAL, HEIGHT, WIDTH
from datasets import PairDataset, load_manifest, load_virtual_bgr
from geometry import verify_trajectory, relative_poses


def make_data(tmp_path, n_images=24):
    """Mock a one-log 20Hz AV2 Sensor train split with real JPEGs and in-memory feather tables."""
    root = tmp_path / 'av2'
    log = root / 'train' / 'test-log-uuid'
    image_dir = log / 'sensors' / 'cameras' / 'stereo_front_left'
    image_dir.mkdir(parents=True)
    (log / 'calibration').mkdir()
    start = 315968112437425433
    # All rows have camera optical->vehicle rotation: x(right)->-Y, y(down)->-Z, z(forward)->X.
    # Quaternion matches approximate AV2 front camera convention.
    q = [0.5, -0.5, 0.5, -0.5]
    camera = dict(sensor_name='stereo_front_left', qw=q[0], qx=q[1], qy=q[2], qz=q[3],
                  tx_m=1.55, ty_m=0.25, tz_m=1.2)
    intrinsic = dict(sensor_name='stereo_front_left', fx_px=700., fy_px=700.,
                     cx_px=320., cy_px=240., width_px=640, height_px=480,
                     k1=.08, k2=-.12, k3=.04)
    imu_rows = []
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[..., 0] = 50
    img[..., 1] = np.linspace(0, 220, 640, dtype=np.uint8)[None]
    img[..., 2] = 160
    for i in range(n_images):
        ts = start + i * 50_000_000
        cv2.imwrite(str(image_dir / f'{ts}.jpg'), img)
        imu_rows.append(dict(timestamp_ns=ts, qw=1.0, qx=0.0, qy=0.0, qz=0.0,
                             tx_m=i * .05, ty_m=.01 * i, tz_m=0.))
    # Offset localization timestamps by 10 ms to force actual interpolation (not lookup).
    local_pose_rows = []
    for i in range(n_images + 1):
        ts = start - 10_000_000 + i * 50_000_000
        local_pose_rows.append(dict(timestamp_ns=ts, qw=1.0, qx=0.0, qy=0.0, qz=0.0,
                                    tx_m=(ts - start) * 1e-9, ty_m=0.0, tz_m=0.0))
    tables = {
        str(log / 'calibration' / 'intrinsics.feather'): pd.DataFrame([intrinsic]),
        str(log / 'calibration' / 'egovehicle_SE3_sensor.feather'): pd.DataFrame([camera]),
        str(log / 'city_SE3_egovehicle.feather'): pd.DataFrame(local_pose_rows),
    }
    return root, tables


def test_av2_index_10hz_pose_and_extrinsic(tmp_path, monkeypatch):
    root, tables = make_data(tmp_path)
    monkeypatch.setattr(av2_loader, '_read_feather', lambda p: tables[str(p)])
    args = Namespace(root=str(root), av2_camera='stereo_front_left', av2_fps=10,
                     max_logs_per_split=0, av2_max_pose_gap_ms=150., strict=True)
    result = av2_loader.index_av2(args)
    assert len(result) == 1
    seq = result[0]
    assert seq['source'] == 'av2' and seq['split'] == 'train'
    assert len(seq['frames']) == 12
    assert seq['distortion'] == [0] * 5  # undistorted JPEG, but lens metadata is preserved
    assert seq['raw_lens_radial_coefficients'] == [.08, -.12, .04]
    dt = np.diff([f['timestamp_s'] for f in seq['frames']])
    np.testing.assert_allclose(dt, 0.1, atol=1e-9)
    cam_to_vehicle = np.array(seq['T_vehicle_camera'])
    for f in seq['frames']:
        np.testing.assert_allclose(f['pose'], np.array(f['vehicle_pose']) @ cam_to_vehicle, atol=1e-10)
    p = np.asarray([f['pose'] for f in seq['frames']])
    verify_trajectory(p)
    # Vehicle +x -> camera +z, relative translation for 100ms should be ~ +0.1m along optical z
    dT = relative_poses(p)
    assert abs(np.linalg.norm(dT[0, :3, 3]) - .1) < 1e-9
    assert dT[0, 2, 3] > 0.09
    out = root / 'indexes' / 'av2.json'
    out.parent.mkdir()
    out.write_text(json.dumps(dict(format_version=3, stage='raw',
                                   camera=dict(K_virtual=K_VIRTUAL.tolist(),
                                               virtual_size=[WIDTH, HEIGHT], virtual_hfov_deg=45.0),
                                   sequences=result)))
    manifest = load_manifest(out)
    data = PairDataset(manifest, split='train', mode='online')
    x, K, label = data[0]
    assert x.shape == (6, HEIGHT, WIDTH)
    assert K.shape == (3, 3) and label.shape == (6,)
    np.testing.assert_allclose(K, K_VIRTUAL.astype(np.float32))
    virtual = load_virtual_bgr(seq, seq['frames'][0], 'online')
    assert virtual.shape == (384, 640, 3)


def test_pose_slerp_is_stable():
    first = dict(timestamp_ns=0, qw=1., qx=0., qy=0., qz=0., tx_m=0., ty_m=0., tz_m=0.)
    q = [np.cos(np.pi/8), 0., 0., np.sin(np.pi/8)]
    second = dict(timestamp_ns=100_000_000, qw=q[0], qx=q[1], qy=q[2], qz=q[3],
                  tx_m=2., ty_m=0., tz_m=0.)
    interp = av2_loader.EgoPoseInterpolator(pd.DataFrame([first, second]))
    T = interp(50_000_000)
    np.testing.assert_allclose(T[:3, 3], [1., 0., 0.])
    assert abs(np.arctan2(T[1, 0], T[0, 0]) - np.pi/8) < 1e-6
    assert interp(200_000_000) is None
