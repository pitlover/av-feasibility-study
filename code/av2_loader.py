"""AV2 Sensor Dataset -> ego_motion_dualmode v3 raw index.

No RGB images are copied here. Source images are already undistorted JPEGs.
We keep their *optical* camera center and orientation unchanged while applying
K_virtual during materialize() or load_virtual_bgr(..., mode='online').

The 20 Hz raw images are sub-sampled to target FPS, and every GT vehicle pose
is interpolated at the EXACT selected image timestamp (not a nearby lidar sweep).
"""
from pathlib import Path
import warnings

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def _read_feather(path):
    # pandas depends on pyarrow for Feather I/O. Lazy import preserves KITTI/Waymo tools.
    try:
        import pandas as pd
        return pd.read_feather(path)
    except ImportError as exc:
        raise RuntimeError(
            'AV2 indexing needs pandas + pyarrow. Run: python -m pip install pandas pyarrow'
        ) from exc


def se3_from_av2_row(row):
    """AV2 uses scalar-first (qw,qx,qy,qz) and meters; returns dst_SE3_src."""
    q_wxyz = np.array([row['qw'], row['qx'], row['qy'], row['qz']], dtype=np.float64)
    xyz = np.array([row['tx_m'], row['ty_m'], row['tz_m']], dtype=np.float64)
    if not np.isfinite(q_wxyz).all() or not np.isfinite(xyz).all():
        raise ValueError('Nonfinite AV2 SE3 values')
    if not np.isclose(np.linalg.norm(q_wxyz), 1.0, atol=1e-2):
        raise ValueError('Invalid AV2 quaternion')
    q_wxyz = q_wxyz / np.linalg.norm(q_wxyz)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = Rotation.from_quat(q_wxyz[[1, 2, 3, 0]]).as_matrix()
    out[:3, 3] = xyz
    return out


class EgoPoseInterpolator:
    """Linearly interpolate translation and SLERP rotation at image timestamps.

    Reject extrapolation, and reject interpolating through localization gaps.
    Nanosecond timestamps stay int64; only differences use floating point seconds.
    """

    def __init__(self, poses_df, max_bracket_gap_ms=150.0):
        required = {'timestamp_ns', 'qw', 'qx', 'qy', 'qz', 'tx_m', 'ty_m', 'tz_m'}
        if not required.issubset(poses_df.columns):
            raise ValueError(f'AV2 pose table missing columns {required - set(poses_df.columns)}')
        df = poses_df.sort_values('timestamp_ns').drop_duplicates('timestamp_ns', keep='first')
        self.ts = df['timestamp_ns'].to_numpy(dtype=np.int64)
        self.poses = [se3_from_av2_row(row) for _, row in df.iterrows()]
        self.max_gap_ns = int(max_bracket_gap_ms * 1e6)
        if len(self.ts) < 2:
            raise ValueError('Not enough AV2 pose timestamps for interpolation')

    def __call__(self, timestamp_ns):
        timestamp_ns = int(timestamp_ns)
        at = int(np.searchsorted(self.ts, timestamp_ns, side='left'))
        if at < len(self.ts) and int(self.ts[at]) == timestamp_ns:
            return self.poses[at].copy()
        if at == 0 or at >= len(self.ts):
            return None  # NEVER extrapolate past available GPS/localization poses
        left = at - 1
        gap = int(self.ts[at]) - int(self.ts[left])
        if gap <= 0 or gap > self.max_gap_ns:
            return None
        fraction = (timestamp_ns - int(self.ts[left])) / gap
        A, B = self.poses[left], self.poses[at]
        output = np.eye(4, dtype=np.float64)
        # SLERP operates on SO(3), unlike linear interpolation of rotation matrices.
        rot = Rotation.from_matrix(np.stack([A[:3, :3], B[:3, :3]]))
        output[:3, :3] = Slerp([0., 1.], rot)([fraction]).as_matrix()[0]
        output[:3, 3] = (1. - fraction) * A[:3, 3] + fraction * B[:3, 3]
        return output


def _camera_calibration(log_dir, camera):
    intrinsic_file = log_dir / 'calibration' / 'intrinsics.feather'
    extrinsic_file = log_dir / 'calibration' / 'egovehicle_SE3_sensor.feather'
    intr = _read_feather(intrinsic_file)
    extr = _read_feather(extrinsic_file)
    intr_rows = intr.loc[intr['sensor_name'] == camera]
    extr_rows = extr.loc[extr['sensor_name'] == camera]
    if len(intr_rows) != 1 or len(extr_rows) != 1:
        raise ValueError(f'{camera} calibration unavailable or duplicated in {log_dir}')
    row = intr_rows.iloc[0]
    fx, fy, cx, cy = [float(row[name]) for name in ('fx_px', 'fy_px', 'cx_px', 'cy_px')]
    if not all(np.isfinite([fx, fy, cx, cy])) or min(fx, fy) <= 0:
        raise ValueError(f'Invalid {camera} intrinsics')
    K = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]], dtype=np.float64)
    T_vehicle_camera = se3_from_av2_row(extr_rows.iloc[0])
    original_radial = [float(row[x]) for x in ('k1', 'k2', 'k3') if x in row.index]
    shape = [int(row['height_px']), int(row['width_px'])] if {'height_px', 'width_px'}.issubset(row.index) else None
    return K, T_vehicle_camera, original_radial, shape


def _split_by_time(frames, prefix, split, camera_name, K_src, T_vehicle_camera,
                   original_radial, source_fps, target_fps, max_gap_factor=1.45):
    groups, run = [], []
    nominal = 1. / target_fps
    for fr in frames + [None]:
        if fr is not None and run:
            dt = fr['timestamp_s'] - run[-1]['timestamp_s']
            if not (nominal * .65 <= dt <= nominal * max_gap_factor):
                if len(run) >= 2:
                    groups.append(run)
                run = []
        if fr is None:
            if len(run) >= 2:
                groups.append(run)
            run = []
        else:
            run.append(fr)
    return [dict(
        name=f'{prefix}_part{i}', source='av2', split=split,
        camera_name=camera_name,
        K_src=K_src.tolist(),
        # AV2 distributed camera JPGs are ALREADY UNDISTORTED; no second correction.
        distortion=[0., 0., 0., 0., 0.],
        raw_lens_radial_coefficients=original_radial,
        source_images_undistorted=True,
        original_fps=source_fps, target_fps=target_fps,
        T_vehicle_camera=T_vehicle_camera.tolist(),
        vehicle_frame='av2_egovehicle',
        extrinsic_source='calibration/egovehicle_SE3_sensor.feather',
        frames=g,
    ) for i, g in enumerate(groups)]


def index_av2(args):
    """Returns v3 manifest sequences; train/val only (not hidden benchmark test)."""
    root = Path(args.root)
    if args.av2_fps not in (10, 20):
        raise ValueError('AV2 target fps must be 10 or 20')
    stride = 20 // args.av2_fps
    all_sequences = []
    for directory, split in [('train', 'train'), ('val', 'val')]:
        split_root = root / directory
        if not split_root.is_dir() and directory == 'val':
            split_root = root / 'validation'
        if not split_root.is_dir() and directory == 'train':
            split_root = root / 'training'
        if not split_root.is_dir():
            warnings.warn(f'Missing AV2 split {directory} at {root}; skipping')
            continue
        logs = sorted(d for d in split_root.iterdir() if d.is_dir())
        if args.max_logs_per_split > 0:
            logs = logs[:args.max_logs_per_split]
        for log_dir in logs:
            try:
                K, T_VC, radial, image_shape = _camera_calibration(log_dir, args.av2_camera)
                pose_file = log_dir / 'city_SE3_egovehicle.feather'
                poses_interp = EgoPoseInterpolator(
                    _read_feather(pose_file), max_bracket_gap_ms=args.av2_max_pose_gap_ms
                )
                folder = log_dir / 'sensors' / 'cameras' / args.av2_camera
                all_paths = sorted(folder.glob('*.jpg'), key=lambda p: int(p.stem))
                if len(all_paths) < 2:
                    raise ValueError(f'Not enough JPEG images in {folder}')
                # Downsample *images*, not just relative action labels. Input becomes 10Hz.
                chosen = all_paths[::stride]
                first_ts = int(chosen[0].stem)
                frames = []
                missing_pose = 0
                for image_path in chosen:
                    ts = int(image_path.stem)
                    T_WV = poses_interp(ts)
                    if T_WV is None:
                        frames.append(None)  # Must not connect across a missing pose.
                        missing_pose += 1
                        continue
                    T_WC = T_WV @ T_VC
                    frames.append(dict(
                        timestamp_ns=ts,
                        timestamp_s=(ts - first_ts) * 1e-9,
                        pose=T_WC.tolist(),
                        vehicle_pose=T_WV.tolist(),
                        raw_path=str(image_path.resolve()),
                    ))
                new_groups = _split_by_time(
                    frames, f'av2_{log_dir.name}_{args.av2_camera}', split,
                    args.av2_camera, K, T_VC, radial, 20, args.av2_fps
                )
                for s in new_groups:
                    if image_shape is not None:
                        s['source_image_size_hw'] = image_shape
                    s['pose_timestamp_method'] = 'SE3 translation interpolation + SO3 SLERP at image timestamp'
                all_sequences.extend(new_groups)
                print(f'AV2 {split} {log_dir.name}: input={len(all_paths)} '
                      f'selected={len(chosen)} groups={len(new_groups)} missing_pose={missing_pose}', flush=True)
            except (FileNotFoundError, ValueError, KeyError, OSError) as error:
                if args.strict:
                    raise RuntimeError(f'AV2 log failed: {log_dir}') from error
                print(f'SKIP AV2 {log_dir}: {error}', flush=True)
    if not all_sequences:
        raise RuntimeError('No usable AV2 sequences. Verify --root, Sensor train/val directories, and feather files.')
    return all_sequences
