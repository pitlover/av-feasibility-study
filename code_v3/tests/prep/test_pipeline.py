import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cosmos3prep.geometry import encode_local, decode_local, interpolate_pose, localize
from cosmos3prep.core import Frame, frame_runs_10hz, frame_runs_kitti_native
from cosmos3prep.cli import main


def test_roundtrip_conventions():
    poses = np.repeat(np.eye(4)[None], 61, axis=0)
    for i, angle in enumerate(np.linspace(0, .4, 61)):
        poses[i, :3, :3] = Rotation.from_euler('z', angle).as_matrix()
        poses[i, :3, 3] = [i * .35, i*.002, 0]
    poses = localize(poses)
    for convention in ('backward_framewise', 'backward_anchored'):
        a = encode_local(poses, convention)
        assert a.shape == (60, 9)
        dec = decode_local(a, convention)
        assert np.allclose(dec, poses, atol=1e-6)


def test_resample_20_to_10():
    f = [Frame(t_s=k*.05, image=b'', world_ego=np.eye(4),
               ego_camera=np.eye(4), K=np.eye(3)) for k in range(121)]
    runs = frame_runs_10hz(f)
    assert len(runs) == 1 and len(runs[0]) == 61
    assert np.allclose([r.t_s for r in runs[0]], np.arange(61) * .1)


def test_pose_interpolation():
    poses = np.stack([np.eye(4), np.eye(4)])
    poses[1,:3,3] = [2, 0, 0]
    poses[1,:3,:3] = Rotation.from_euler('z', 90, degrees=True).as_matrix()
    t = interpolate_pose(np.array([0, 100_000_000]), poses, 50_000_000)
    assert np.allclose(t[:3,3], [1,0,0])
    assert np.allclose(Rotation.from_matrix(t[:3,:3]).as_euler('xyz', degrees=True)[2], 45)


def test_kitti_end_to_end(tmp_path):
    root = tmp_path / 'kitti'
    seq = root / 'sequences' / '00'
    (seq / 'image_2').mkdir(parents=True)
    (root / 'poses').mkdir(parents=True)
    k = np.array([[90.,0,48],[0,90.,32],[0,0,1]])
    p0 = np.c_[k, [0,0,0]]
    p2 = np.c_[k, [-9.,0,0]]
    tr = np.array([[0,-1,0,0],[0,0,-1,0],[1,0,0,0]], np.float64)
    (seq / 'calib.txt').write_text(
       '\n'.join([f'P0: {" ".join(map(str,p0.ravel()))}',
                  f'P2: {" ".join(map(str,p2.ravel()))}',
                  f'Tr: {" ".join(map(str,tr.ravel()))}'])+'\n')
    poses = []
    for i in range(61):
        img = np.zeros((64,96,3), np.uint8)
        img[:,:,0] = i*4
        assert cv2.imwrite(str(seq / 'image_2' / f'{i:06d}.png'), img)
        pose = np.eye(4)
        pose[2,3] = i * .2
        poses.append(pose[:3].reshape(-1))
    np.savetxt(seq / 'times.txt', np.arange(61) * .1, fmt='%.9f')
    np.savetxt(root / 'poses' / '00.txt', np.stack(poses), fmt='%.9f')
    out = tmp_path / 'out'
    main(['kitti','--root',str(root),'--out',str(out),'--sequences','00',
          '--encoder','local','--width','128','--height','72', '--max-clips','1', '--strict'])
    metas = list(out.rglob('metadata.json'))
    assert len(metas) == 1
    d = json.loads(metas[0].read_text())
    assert d['frame_count'] == 61 and d['action_count'] == 60
    assert d['pose_frame'] == 'velodyne'
    T = np.load(metas[0].parent/'T_ego_camera.npy')
    assert T.shape == (61, 4,4)
    main(['validate', '--out', str(out)])


def test_av2_official_sensor_layout_20hz_to_10hz(tmp_path, monkeypatch):
    """Simulate AV2's official Feather schemas + timestamped JPEG folder."""
    import pandas as pd
    root = tmp_path / 'sensor' / 'train'
    log = root / 'fake-log-uuid'
    img_dir = log / 'sensors' / 'cameras' / 'stereo_front_left'
    img_dir.mkdir(parents=True)
    (log / 'calibration').mkdir()
    for f in ['city_SE3_egovehicle.feather', 'calibration/egovehicle_SE3_sensor.feather',
              'calibration/intrinsics.feather']:
        (log / f).touch()
    t0 = 315968112437425433
    img = np.full((48, 96, 3), 127, np.uint8)
    for i in range(121):
        assert cv2.imwrite(str(img_dir / f'{t0 + i * 50_000_000}.jpg'), img)
    gt = pd.DataFrame({
        'timestamp_ns': [t0 + i * 100_000_000 for i in range(61)],
        'qw': [1.] * 61, 'qx': [0.] * 61, 'qy': [0.] * 61, 'qz': [0.] * 61,
        'tx_m': np.arange(61) * 0.5, 'ty_m': [0.] * 61, 'tz_m': [0.] * 61,
    })
    extr = pd.DataFrame([dict(sensor_name='stereo_front_left', qw=1., qx=0., qy=0., qz=0.,
                              tx_m=1.5, ty_m=0.2, tz_m=1.2)])
    intr = pd.DataFrame([dict(sensor_name='stereo_front_left', fx_px=500., fy_px=500.,
                              cx_px=48., cy_px=24.)])
    def fake_read_feather(p):
        name = Path(p).name
        return {'city_SE3_egovehicle.feather': gt, 'egovehicle_SE3_sensor.feather': extr,
                'intrinsics.feather': intr}[name]
    monkeypatch.setattr(pd, 'read_feather', fake_read_feather)
    out = tmp_path / 'out_av2'
    main(['av2', '--root', str(root), '--out', str(out), '--encoder', 'local',
          '--max-clips', '1', '--width', '128', '--height', '72', '--strict'])
    metas = list(out.rglob('metadata.json'))
    assert len(metas) == 1
    t = np.load(metas[0].parent / 'timestamps_s.npy')
    assert len(t) == 61
    assert np.allclose(np.diff(t), .1, atol=1e-6)
    pose = np.load(metas[0].parent / 'poses_local_ego.npy')
    assert np.allclose(pose[-1, 0, 3], 30.)
    main(['validate', '--out', str(out)])


def test_waymo_v1_proto_adapter_with_mock_runtime(tmp_path, monkeypatch):
    """Mock protobuf field layout; does not replace test on real Waymo TFRecords."""
    import sys
    import types
    class FakeRaw:
        def __init__(self, idx): self.idx = idx
        def numpy(self): return str(self.idx).encode()
    class FakeCameraName:
        FRONT = 1
    class FakeTransform:
        def __init__(self, m): self.transform = list(np.asarray(m).reshape(-1))
    class FakeProtoFrame:
        def ParseFromString(self, blob):
            idx = int(blob.decode())
            self.timestamp_micros = 1_500_000_000_000_000 + idx * 100_000
            self.context = types.SimpleNamespace(name='test_waymo_segment', camera_calibrations=[
                types.SimpleNamespace(name=1, intrinsic=[90., 90., 48., 32.,0,0,0,0,0],
                                      extrinsic=FakeTransform(np.eye(4)))])
            T = np.eye(4)
            T[0, 3] = idx * .3
            image = np.ones((64,96,3), np.uint8) * 110
            valid, buf = cv2.imencode('.jpg', image)
            assert valid
            self.images = [types.SimpleNamespace(name=1, image=buf.tobytes(),
                          pose=FakeTransform(T), pose_timestamp=1_500_000_000 + idx * .1)]
    fake_tf = types.SimpleNamespace(data=types.SimpleNamespace(
        TFRecordDataset=lambda files, compression_type='': [FakeRaw(i) for i in range(61)]))
    fake_proto = types.SimpleNamespace(Frame=FakeProtoFrame, CameraName=FakeCameraName)
    fake_waymo = types.ModuleType('waymo_open_dataset')
    fake_waymo.dataset_pb2 = fake_proto
    monkeypatch.setitem(sys.modules, 'tensorflow', fake_tf)
    monkeypatch.setitem(sys.modules, 'waymo_open_dataset', fake_waymo)
    root = tmp_path / 'waymo' / 'training'
    root.mkdir(parents=True)
    (root / 'test.tfrecord').touch()
    out = tmp_path / 'out_waymo'
    main(['waymo','--root', str(root), '--out', str(out), '--max-clips','1',
          '--encoder','local','--width','128','--height','72','--strict'])
    meta = next(out.rglob('metadata.json'))
    d = json.loads(meta.read_text())
    assert d['pose_frame'] == 'vehicle' and d['camera'] == 'FRONT'
    pose = np.load(meta.parent / 'poses_local_ego.npy')
    assert np.allclose(pose[-1, 0, 3], 18, atol=1e-4)
    main(['validate','--out', str(out)])


def test_kitti_separate_pose_directory(tmp_path):
    """Shared KITTI images need not contain poses/; GT comes from another root."""
    root = tmp_path / 'shared' / 'dataset'
    seq = root / 'sequences' / '00'
    imgdir = seq / 'image_2'
    imgdir.mkdir(parents=True)
    poses_dir = tmp_path / 'private' / 'data_odometry_poses' / 'dataset' / 'poses'
    poses_dir.mkdir(parents=True)

    K = np.array([[90., 0., 48.], [0., 90., 32.], [0., 0., 1.]])
    P0 = np.c_[K, np.zeros(3)]
    P2 = np.c_[K, [-9., 0., 0.]]
    Tr = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0]])
    (seq / 'calib.txt').write_text('\n'.join([
        f'P0: {" ".join(map(str, P0.ravel()))}',
        f'P2: {" ".join(map(str, P2.ravel()))}',
        f'Tr: {" ".join(map(str, Tr.ravel()))}',
    ]) + '\n')
    np.savetxt(seq / 'times.txt', np.arange(61) * .1)
    pose_rows = []
    for i in range(61):
        assert cv2.imwrite(str(imgdir / f'{i:06d}.png'), np.full((64, 96, 3), 90, np.uint8))
        T = np.eye(4)
        T[2, 3] = i * .2
        pose_rows.append(T[:3].ravel())
    np.savetxt(poses_dir / '00.txt', pose_rows)
    assert not (root / 'poses').exists()

    output = tmp_path / 'result'
    main(['kitti', '--root', str(root), '--poses-dir', str(poses_dir),
          '--out', str(output), '--sequences', '00', '--max-clips', '1',
          '--encoder', 'local', '--width', '128', '--height', '72', '--strict'])
    assert len(list(output.rglob('metadata.json'))) == 1
    main(['validate', '--out', str(output)])
    assert not (root / 'poses').exists()


def test_kitti_separate_pose_and_calibration_roots(tmp_path):
    """Shared images remain untouched; private GT and full calibration are read separately."""
    shared = tmp_path / 'team_shared' / 'dataset'
    seq = shared / 'sequences' / '00'
    imgdir = seq / 'image_2'
    imgdir.mkdir(parents=True)
    private_pose = tmp_path / 'personal' / 'data_odometry_poses' / 'dataset' / 'poses'
    private_pose.mkdir(parents=True)
    private_calib = tmp_path / 'personal' / 'data_odometry_calib' / 'dataset'
    full_calib_dir = private_calib / 'sequences' / '00'
    full_calib_dir.mkdir(parents=True)

    K = np.array([[90., 0., 48.], [0., 90., 32.], [0., 0., 1.]])
    P2 = np.c_[K, [-9., 0., 0.]]
    Tr = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0]])
    # Team's calibration is incomplete (only P0-P3), as in the reported failure.
    (seq / 'calib.txt').write_text(f'P2: {" ".join(map(str, P2.ravel()))}\n')
    original_shared_calib = (seq / 'calib.txt').read_bytes()
    # Downloaded official calibration is complete, with Tr.
    full_calib_path = full_calib_dir / 'calib.txt'
    full_calib_path.write_text('\n'.join([
        f'P2: {" ".join(map(str, P2.ravel()))}',
        f'Tr: {" ".join(map(str, Tr.ravel()))}',
    ]) + '\n')
    np.savetxt(seq / 'times.txt', np.arange(61) * .1)
    gt_rows = []
    for i in range(61):
        assert cv2.imwrite(str(imgdir / f'{i:06d}.png'),
                           np.full((64, 96, 3), 120, dtype=np.uint8))
        T = np.eye(4)
        T[2, 3] = i * 0.3
        gt_rows.append(T[:3, :].ravel())
    np.savetxt(private_pose / '00.txt', gt_rows)
    out = tmp_path / 'processed'
    main(['kitti', '--root', str(shared), '--poses-dir', str(private_pose),
          '--calib-root', str(private_calib), '--sequences', '00',
          '--camera', 'image_2', '--out', str(out), '--max-clips', '1',
          '--encoder', 'local', '--width', '128', '--height', '72', '--strict'])
    main(['validate', '--out', str(out)])
    assert len(list(out.rglob('metadata.json'))) == 1
    assert (seq / 'calib.txt').read_bytes() == original_shared_calib
    assert not (shared / 'poses').exists()


def test_resolve_kitti_calibration_multiple_private_layouts(tmp_path):
    from cosmos3prep.datasets import _resolve_kitti_calib
    seq_dir = tmp_path / 'shared' / 'sequences' / '00'
    seq_dir.mkdir(parents=True)
    archive = tmp_path / 'data_odometry_calib'
    full_calib = archive / 'dataset' / 'sequences' / '00' / 'calib.txt'
    full_calib.parent.mkdir(parents=True)
    full_calib.write_text('Tr: 1 0 0 0 0 1 0 0 0 0 1 0\n')
    for supplied in (archive, archive / 'dataset', archive / 'dataset' / 'sequences',
                     archive / 'dataset' / 'sequences' / '00', full_calib):
        assert _resolve_kitti_calib(seq_dir, '00', supplied) == full_calib


def test_kitti_nominal_10hz_103736ms_timestamp_drift(tmp_path):
    """61 sequential KITTI images should not be lost to a global 0.1-s grid."""
    root = tmp_path / 'shared_dataset'
    seq = root / 'sequences' / '00'
    imgdir = seq / 'image_2'
    imgdir.mkdir(parents=True)
    posesdir = tmp_path / 'private_poses'
    posesdir.mkdir()
    calib = tmp_path / 'private_calib' / 'sequences' / '00'
    calib.mkdir(parents=True)
    K = np.array([[90., 0., 48.], [0., 90., 32.], [0., 0., 1.]])
    P2 = np.c_[K, [-9., 0., 0.]]
    Tr = np.array([[0., -1., 0., 0.], [0., 0., -1., 0.], [1., 0., 0., 0.]])
    (calib / 'calib.txt').write_text(
        f'P2: {" ".join(map(str, P2.ravel()))}\n' +
        f'Tr: {" ".join(map(str, Tr.ravel()))}\n')
    ts = np.arange(61) * 0.103736
    np.savetxt(seq / 'times.txt', ts, fmt='%.9f')
    gt = []
    frames = []
    for i in range(61):
        im = np.full((48, 96, 3), i, dtype=np.uint8)
        cv2.imwrite(str(imgdir / f'{i:06d}.png'), im)
        T = np.eye(4)
        T[2, 3] = i * .3
        gt.append(T[:3].reshape(-1))
        frames.append(Frame(t_s=float(ts[i]), image=imgdir / f'{i:06d}.png',
                            world_ego=T, ego_camera=np.eye(4), K=K,
                            source_id=f'00/{i:06d}.png'))
    np.savetxt(posesdir / '00.txt', gt)
    # Old strict-grid sampling does not preserve a 61-frame run.
    assert max(map(len, frame_runs_10hz(frames)), default=0) < 61
    assert max(map(len, frame_runs_kitti_native(frames))) == 61
    out = tmp_path / 'processed'
    main(['kitti', '--root', str(root), '--poses-dir', str(posesdir),
          '--calib-root', str(calib.parent.parent), '--sequences', '00',
          '--out', str(out), '--max-clips', '1', '--encoder', 'local',
          '--width', '128', '--height', '72', '--strict'])
    meta = next(out.rglob('metadata.json'))
    d = json.loads(meta.read_text())
    assert abs(d['actual_capture_duration_s'] - 6.22416) < .0001
    assert d['clip_timebase'] == 'kitti_native_nominal_10fps'
    main(['validate', '--out', str(out)])


def test_kitti_native_splits_large_timestamp_and_image_gaps():
    def f(t, i):
        return Frame(t_s=t, image=b'', world_ego=np.eye(4),
                     ego_camera=np.eye(4), K=np.eye(3), source_id=f'00/{i:06d}.png')
    frames = [f(0, 0), f(.104, 1), f(.208, 2), f(.9, 3), f(1.004, 5)]
    assert [len(r) for r in frame_runs_kitti_native(frames)] == [3, 1, 1]


def test_h264_codec_overwrite_and_deduplicated_index(tmp_path):
    """Old MP4V output is re-encoded as H.264, with one index entry per clip."""
    import shutil
    import subprocess
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg/ffprobe not available')

    root = tmp_path / 'public' / 'dataset'
    seq = root / 'sequences' / '00'
    imgdir = seq / 'image_2'
    imgdir.mkdir(parents=True)
    gt_dir = tmp_path / 'private_poses'
    gt_dir.mkdir()
    calib_root = tmp_path / 'private_calib'
    (calib_root / 'sequences' / '00').mkdir(parents=True)

    K = np.array([[90., 0., 48.], [0., 90., 32.], [0., 0., 1.]])
    P2 = np.c_[K, [-9., 0., 0.]]
    Tr = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0]])
    (calib_root / 'sequences' / '00' / 'calib.txt').write_text('\n'.join([
        f'P2: {" ".join(map(str, P2.ravel()))}',
        f'Tr: {" ".join(map(str, Tr.ravel()))}',
    ]))
    poses = []
    for i in range(121):
        img = np.ones((64, 96, 3), np.uint8) * (i % 255)
        assert cv2.imwrite(str(imgdir / f'{i:06d}.png'), img)
        T = np.eye(4)
        T[2, 3] = i * .2
        poses.append(T[:3].reshape(-1))
    np.savetxt(gt_dir / '00.txt', poses)
    np.savetxt(seq / 'times.txt', np.arange(121) * .103736)
    output = tmp_path / 'processed'
    common = ['kitti', '--root', str(root), '--poses-dir', str(gt_dir),
              '--calib-root', str(calib_root), '--out', str(output),
              '--sequences', '00', '--encoder', 'local', '--width', '128',
              '--height', '72', '--strict']
    main(common + ['--video-codec', 'mp4v', '--max-clips', '1'])
    video = output / 'kitti' / 'train' / 'kitti_00_000000' / 'video.mp4'
    assert video.exists()

    def codec():
        return subprocess.check_output([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name,pix_fmt',
            '-of', 'default=noprint_wrappers=1', str(video)
        ], text=True)

    assert 'codec_name=mpeg4' in codec()
    main(common + ['--max-clips', '1', '--overwrite'])
    assert 'codec_name=h264' in codec()
    assert 'pix_fmt=yuv420p' in codec()
    meta = json.loads((video.parent / 'metadata.json').read_text())
    assert meta['video_codec'] == 'h264'
    main(['validate', '--out', str(output)])
    assert len((output / 'index.jsonl').read_text().splitlines()) == 1
    # Runs on *all* clips, skipping existing 000000 and creating 000060.
    main(common)
    assert len(list(output.rglob('metadata.json'))) == 2
    assert len((output / 'index.jsonl').read_text().splitlines()) == 2
    # Re-running without overwrite must not grow the index.
    main(common)
    assert len((output / 'index.jsonl').read_text().splitlines()) == 2
    main(['validate', '--out', str(output)])
