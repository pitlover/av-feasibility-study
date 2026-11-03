"""KITTI Odometry C2 optical -> Velodyne or IMU reference extrinsics.

Definitions:
  T_A_B maps B-frame coordinates to A-frame coordinates.
  Odometry calib.txt 'Tr' maps Velodyne -> rectified Cam0.
  Odometry poses/XX.txt provides T_W_C0.
  Camera 2 is a rectified optical frame; image_2 was already rectified.

'velodyne_proxy' is a rigid sensor origin, NOT the true vehicle center.
If an IMU calibration file is available for the matching RAW recording date,
use --kitti-frame imu and --kitti-imu-calib-root to obtain IMU->vehicle proxy.
The IMU center is still not necessarily the center of the vehicle's rear axle.
"""
from pathlib import Path
import numpy as np

from geometry import transform, kitti_cam2_from_pose_cam0

# KITTI Odometry 00..10 -> original KITTI RAW calibration date.
KITTI_ODOM_DATES = {
    '00': '2011_10_03', '01': '2011_10_03', '02': '2011_10_03',
    '03': '2011_09_26', '04': '2011_09_30', '05': '2011_09_30',
    '06': '2011_09_30', '07': '2011_09_30', '08': '2011_09_30',
    '09': '2011_09_30', '10': '2011_09_30',
}


def read_calib(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        if ':' in line:
            name, text = line.split(':', 1)
            values[name.strip()] = np.fromstring(text, sep=' ', dtype=np.float64)
    return values


def _rigid(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def _get_odom_T_C0_Velo(odometry_calib):
    """Odometry 'Tr' is normally 12 values: Velodyne -> rectified Cam0."""
    for name in ('Tr', 'Tr_velo_to_cam'):
        if name in odometry_calib:
            data = odometry_calib[name]
            if data.size != 12:
                raise ValueError(f'KITTI {name} should have 12 values, got {data.size}')
            return transform(data.reshape(3, 4))
    raise ValueError('KITTI odometry calib.txt is missing Tr (Velodyne->Cam0). '
                     'Download the odometry calibration archive or inspect calib.txt.')


def _T_C0_C2(P0, P2):
    """C2 optical -> rectified C0 optical (shared parallel rectified axes)."""
    # -inv(K_i) P_i[:,3] is the camera center in the common rectified frame.
    C0 = -np.linalg.solve(P0[:, :3], P0[:, 3])
    C2 = -np.linalg.solve(P2[:, :3], P2[:, 3])
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = C2 - C0
    return T


def resolve_imu_calibration(root, seq):
    """Allow root/{date}/calib_imu_to_velo.txt or root/{seq}/... ."""
    if root is None:
        raise ValueError('To use --kitti-frame imu, provide --kitti-imu-calib-root.')
    root = Path(root)
    date = KITTI_ODOM_DATES[seq]
    candidates = [root / date / 'calib_imu_to_velo.txt',
                  root / seq / 'calib_imu_to_velo.txt',
                  root / 'calib_imu_to_velo.txt']
    matches = [p for p in candidates if p.is_file()]
    if not matches:
        raise FileNotFoundError(f'KITTI sequence {seq}: IMU->Velodyne calibration not found. '
                                f'Searched: {[str(p) for p in candidates]}')
    return matches[0]


def kitti_T_V_C2(calib, seq, vehicle_frame='velodyne', imu_calib_root=None):
    """Return camera2 optical -> reference transform and metadata.

    vehicle_frame='velodyne': reference is LiDAR, i.e. Velodyne proxy, NOT vehicle center.
    vehicle_frame='imu': reference is KITTI RAW IMU origin, with per-date calibration.
    """
    P0 = calib['P0'].reshape(3, 4)
    P2 = calib['P2'].reshape(3, 4)
    T_C0_C2 = _T_C0_C2(P0, P2)
    T_C0_Velo = _get_odom_T_C0_Velo(calib)
    T_C2_Velo = np.linalg.inv(T_C0_C2) @ T_C0_Velo
    T_Velo_C2 = np.linalg.inv(T_C2_Velo)
    meta = dict(vehicle_frame='velodyne_proxy',
                vehicle_frame_note='LiDAR origin, not physical ego center',
                extrinsic_source='odometry/calib.txt:Tr,P0,P2')
    if vehicle_frame == 'velodyne':
        return T_Velo_C2, meta
    if vehicle_frame != 'imu':
        raise ValueError(f'Unknown KITTI vehicle frame {vehicle_frame}')
    path = resolve_imu_calibration(imu_calib_root, seq)
    raw = read_calib(path)
    if 'R' not in raw or 'T' not in raw:
        raise ValueError(f'IMU calibration file {path} must contain R and T')
    T_Velo_Imu = _rigid(raw['R'], raw['T'])
    T_Imu_C2 = np.linalg.inv(T_Velo_Imu) @ T_Velo_C2
    meta.update(vehicle_frame='imu',
                vehicle_frame_note='KITTI IMU sensor origin, not necessarily vehicle rear axle',
                extrinsic_source=str(path.resolve()))
    return T_Imu_C2, meta


def camera_to_vehicle_pose(T_W_C, T_V_C):
    """Return global vehicle-reference pose T_W_V from global camera pose."""
    return transform(T_W_C) @ np.linalg.inv(transform(T_V_C))
