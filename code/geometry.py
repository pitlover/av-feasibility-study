"""Coordinate conventions: T_WC maps camera optical coordinates to world."""
import numpy as np
from scipy.spatial.transform import Rotation

# Waymo camera calibration coordinates: forward, left, up.
# Camera optical coordinates: right, down, forward.
R_WAYMO_CAM_FROM_OPT = np.array([[0., 0., 1.],
                                  [-1., 0., 0.],
                                  [0., -1., 0.]])


def transform(A):
    a = np.asarray(A, dtype=np.float64)
    if a.shape == (4, 4):
        return a.copy()
    if a.shape == (3, 4):
        T = np.eye(4, dtype=np.float64)
        T[:3, :4] = a
        return T
    raise ValueError(f"Not a valid transform shape: {a.shape}")


def kitti_cam2_from_pose_cam0(pose_cam0, P0, P2):
    """KITTI odometry rectified cam0 to rectified cam2.

    Projection P_i=K_i[I | -C_i], where C_i is camera center in cam0.
    The transform from cam2 to cam0 thus translates by C2-C0.
    """
    C0 = -np.linalg.solve(P0[:, :3], P0[:, 3])
    C2 = -np.linalg.solve(P2[:, :3], P2[:, 3])
    T_C0_C2 = np.eye(4)
    T_C0_C2[:3, 3] = C2 - C0
    return transform(pose_cam0) @ T_C0_C2


def waymo_optical_pose(T_WV, T_VCam):
    T_CamOpt = np.eye(4)
    T_CamOpt[:3, :3] = R_WAYMO_CAM_FROM_OPT
    return transform(T_WV) @ transform(T_VCam) @ T_CamOpt


def relative_poses(global_poses):
    p = np.asarray(global_poses, dtype=np.float64)
    return np.linalg.inv(p[:-1]) @ p[1:]


def action_from_transform(T):
    T = transform(T)
    return np.r_[T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_rotvec()].astype(np.float32)


def transform_from_action(a):
    a = np.asarray(a, dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(a[3:6]).as_matrix()
    T[:3, 3] = a[:3]
    return T


def accumulate(relative):
    trajectory = [np.eye(4)]
    for d in relative:
        trajectory.append(trajectory[-1] @ transform(d))
    return np.stack(trajectory)


def verify_trajectory(poses, tolerance=1e-5):
    p = np.asarray(poses, dtype=np.float64)
    if len(p) < 2:
        raise ValueError('Need at least two poses')
    local = np.linalg.inv(p[0]) @ p
    reconstructed = accumulate(relative_poses(p))
    t_error = np.max(np.linalg.norm(local[:, :3, 3] - reconstructed[:, :3, 3], axis=1))
    r_error = max(Rotation.from_matrix(local[i, :3, :3].T @ reconstructed[i, :3, :3]).magnitude()
                  for i in range(len(p)))
    if max(t_error, r_error) > tolerance:
        raise ValueError(f"GT reconstruction failed: translation={t_error:.3g}, rotation={r_error:.3g}")
    return t_error, r_error
