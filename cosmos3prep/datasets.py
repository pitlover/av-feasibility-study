"""Adapters for official KITTI Odometry, Waymo Perception v1, AV2 Sensor data."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .core import Frame
from .geometry import matrix4, from_wxyz_row, interpolate_pose


def _split_from_path(p: Path) -> str:
    parents = [s.lower() for s in p.parts]
    if 'train' in parents or 'training' in parents:
        return 'train'
    if 'val' in parents or 'validation' in parents:
        return 'val'
    if 'test' in parents or 'testing' in parents:
        return 'test'
    return 'unspecified'


def _read_calib_txt(path: Path) -> dict[str, np.ndarray]:
    c = {}
    for line in path.read_text().splitlines():
        if ':' not in line:
            continue
        k, v = line.split(':', 1)
        c[k.strip()] = np.fromstring(v, sep=' ', dtype=np.float64)
    return c



def _resolve_kitti_calib(seq_path: Path, seq: str, calib_root: Path | None) -> Path:
    """Resolve official odometry calibration without changing shared dataset files.

    The official calibration archive commonly extracts to
    <archive>/dataset/sequences/<seq>/calib.txt. Some mirrors distribute
    calib_<seq>.txt instead; support that layout as well.
    """
    if calib_root is None:
        return seq_path / 'calib.txt'
    base = Path(calib_root).expanduser()
    if base.is_file():
        if base.name != 'calib.txt' or seq != seq_path.name:
            raise ValueError(f'Use a calibration directory for multiple KITTI sequences: {base}')
        return base
    candidates = (
        base / 'sequences' / seq / 'calib.txt',
        base / 'dataset' / 'sequences' / seq / 'calib.txt',
        base / seq / 'calib.txt',
        base / f'calib_{seq}.txt',
        base / 'calib.txt',  # for a single-sequence calibration directory
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f'KITTI {seq}: no calibration file found under --calib-root={base}. '
        'Expected dataset/sequences/00/calib.txt (or sequences/00/calib.txt).'
    )

def iter_kitti(root: Path, camera: str = 'image_2', sequences: list[str] | None = None,
               poses_dir: Path | None = None, calib_root: Path | None = None):
    """KITTI Odometry images under root/sequences/; GT may be in separate poses_dir.

    'Tr' maps Velodyne -> rectified cam0; GT poses map cam0 -> world.
    Body frame for this pipeline is Velodyne (x forward, y left, z up).
    """
    root = Path(root)
    folder = root / 'sequences'
    poses_dir = Path(poses_dir) if poses_dir is not None else root / 'poses'
    if not poses_dir.is_dir():
        raise FileNotFoundError(
            f'KITTI GT poses directory not found: {poses_dir} '
            '(use --poses-dir /path/to/poses to set it separately)'
        )
    if not folder.is_dir():
        raise FileNotFoundError(f'Not found: {folder} (point --root to KITTI odometry dataset/)')
    ids = sequences or [p.name for p in sorted(folder.iterdir()) if p.is_dir()]
    cam_to_p = {'image_0': 'P0', 'image_1': 'P1', 'image_2': 'P2', 'image_3': 'P3'}
    if camera not in cam_to_p:
        raise ValueError(f'Invalid KITTI camera: {camera}')
    for seq in ids:
        seq_path = folder / seq
        pose_file = poses_dir / f'{seq}.txt'
        if not pose_file.is_file():
            print(f'[skip] KITTI sequence {seq}: no public GT file {pose_file}')
            continue
        calib_path = _resolve_kitti_calib(seq_path, seq, calib_root)
        c = _read_calib_txt(calib_path)
        required = [cam_to_p[camera], 'Tr']
        missing = [k for k in required if k not in c]
        if missing:
            raise ValueError(
                f'KITTI {seq}: calibration file {calib_path} is missing {missing}; '
                f'available keys={list(c)}. This is commonly the partial '
                'calib.txt bundled with image archives. Download official '
                'data_odometry_calib.zip, extract it privately and pass '
                '--calib-root /path/to/data_odometry_calib/dataset.'
            )
        if c['Tr'].size != 12 or c[cam_to_p[camera]].size != 12:
            raise ValueError(f'KITTI {seq}: malformed calibration matrices in {calib_path}')
        print(f'[calibration] KITTI {seq}: {calib_path}')
        T_cam0_velo = matrix4(c['Tr'])
        P = c[cam_to_p[camera]].reshape(3, 4)
        K = P[:, :3]
        # The rectified camera projection matrices encode optical-center offsets.
        # T_camX_cam0: cam0 rectified coordinates -> selected rectified camera.
        T_camX_cam0 = np.eye(4)
        T_camX_cam0[:3, 3] = np.linalg.solve(K, P[:, 3])
        T_camX_velo = T_camX_cam0 @ T_cam0_velo
        T_velo_camX = np.linalg.inv(T_camX_velo)
        pose_table = np.loadtxt(pose_file, dtype=np.float64).reshape(-1, 3, 4)
        world_cam0 = np.stack([matrix4(r) for r in pose_table])
        world_velo = world_cam0 @ T_cam0_velo
        image_dir = seq_path / camera
        images = sorted(image_dir.glob('*.png'))
        times_path = seq_path / 'times.txt'
        if not times_path.exists():
            raise FileNotFoundError(times_path)
        times = np.loadtxt(times_path, dtype=np.float64).reshape(-1)
        frames = []
        for img in images:
            idx = int(img.stem)
            if idx >= len(times) or idx >= len(world_velo):
                continue
            frames.append(Frame(
                t_s=float(times[idx]), image=img, world_ego=world_velo[idx],
                ego_camera=T_velo_camX, K=K, source_id=f'{seq}/{img.name}',
            ))
        yield seq, ('val' if seq in ('09', '10') else 'train'), frames


def iter_av2(root: Path, camera: str = 'stereo_front_left', max_logs: int = 0):
    """AV2 Sensor logs from root/[train|val]/<log_uuid>/sensors/cameras/<camera>/."""
    import pandas as pd

    root = Path(root)
    # Accept either root/train/log or root/log.
    logs = sorted({p.parent for p in root.rglob('city_SE3_egovehicle.feather')})
    if max_logs:
        logs = logs[:max_logs]
    if not logs:
        raise FileNotFoundError(f'No AV2 Sensor log found under {root}')
    for log in logs:
        images_dir = log / 'sensors' / 'cameras' / camera
        if not images_dir.is_dir():
            print(f'[skip] AV2 {log.name}: missing {images_dir}')
            continue
        extr_path = log / 'calibration' / 'egovehicle_SE3_sensor.feather'
        intr_path = log / 'calibration' / 'intrinsics.feather'
        if not extr_path.exists() or not intr_path.exists():
            print(f'[skip] AV2 {log.name}: calibration files missing')
            continue
        extr = pd.read_feather(extr_path)
        intr = pd.read_feather(intr_path)
        ce = extr.loc[extr.sensor_name == camera]
        ci = intr.loc[intr.sensor_name == camera]
        if ce.empty or ci.empty:
            print(f'[skip] AV2 {log.name}: no calibration for {camera}')
            continue
        T_ego_camera = from_wxyz_row(ce.iloc[0])
        row = ci.iloc[0]
        K = np.array([[row.fx_px, 0., row.cx_px],
                      [0., row.fy_px, row.cy_px],
                      [0., 0., 1.]], dtype=np.float64)
        gt = (pd.read_feather(log / 'city_SE3_egovehicle.feather')
              .sort_values('timestamp_ns').drop_duplicates('timestamp_ns'))
        gt_times = gt['timestamp_ns'].to_numpy(dtype=np.int64)
        gt_poses = np.stack([from_wxyz_row(r) for _, r in gt.iterrows()])
        imgs = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')],
                      key=lambda p: int(p.stem))
        frames = []
        for img in imgs:
            t_ns = int(img.stem)
            T_world_ego = interpolate_pose(gt_times, gt_poses, t_ns)
            if T_world_ego is None:
                continue
            frames.append(Frame(
                t_s=t_ns / 1e9, image=img, world_ego=T_world_ego,
                ego_camera=T_ego_camera, K=K, source_id=f'{log.name}/{camera}/{img.name}',
            ))
        yield log.name, _split_from_path(log), frames


def iter_waymo(root: Path, camera: str = 'FRONT', max_logs: int = 0,
               compression: str = ''):
    """Waymo Perception v1 *TFRecord* (not Waymo Motion and not Perception v2 Parquet)."""
    try:
        import tensorflow as tf
        from waymo_open_dataset import dataset_pb2
    except ImportError as e:
        raise RuntimeError(
            'Waymo v1 requires compatible tensorflow + waymo-open-dataset packages. '
            'Use the Waymo-supported Python/TensorFlow versions; do not mix with v2 Parquet.'
        ) from e
    if camera.upper() != 'FRONT':
        raise ValueError('This initial adapter supports Waymo FRONT only')
    root = Path(root)
    files = sorted(p for p in root.rglob('*.tfrecord*') if p.is_file())
    if max_logs:
        files = files[:max_logs]
    if not files:
        raise FileNotFoundError(f'No Waymo v1 TFRecord found under {root}')
    for file in files:
        frames = []
        name = file.stem
        dataset = tf.data.TFRecordDataset([str(file)], compression_type=compression)
        for raw in dataset:
            f = dataset_pb2.Frame()
            f.ParseFromString(bytes(raw.numpy()))
            if f.context.name:
                name = f.context.name
            cal = next((c for c in f.context.camera_calibrations
                        if c.name == dataset_pb2.CameraName.FRONT), None)
            cam = next((c for c in f.images
                        if c.name == dataset_pb2.CameraName.FRONT), None)
            if cal is None or cam is None or len(cam.pose.transform) != 16:
                continue
            # Important: CameraImage.pose is vehicle -> world at pose_timestamp,
            # NOT necessarily Frame.pose at timestamp_micros.
            t_s = float(cam.pose_timestamp) if cam.pose_timestamp > 0 else f.timestamp_micros / 1e6
            if len(cal.extrinsic.transform) != 16 or len(cal.intrinsic) < 4:
                continue
            k = list(cal.intrinsic)
            K = np.array([[k[0], 0., k[2]], [0., k[1], k[3]], [0., 0., 1.]], np.float64)
            # Waymo camera distortion: [k1,k2,p1,p2,k3]
            distortion = np.array(k[4:9] if len(k) >= 9 else [0.] * 5, np.float64)
            frames.append(Frame(
                t_s=t_s, image=bytes(cam.image), world_ego=matrix4(cam.pose.transform),
                ego_camera=matrix4(cal.extrinsic.transform), K=K, distortion=distortion,
                needs_undistort=True,
                source_id=f'{name}/{f.timestamp_micros}/FRONT',
            ))
        yield name, _split_from_path(file), frames
