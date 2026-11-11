"""Virtual camera with SAME camera optical center and orientation.

The image operation changes intrinsics (and undistorts); it does NOT rotate or
translate the camera. T_ego_camera remains the original physical extrinsic.
Coordinate convention: camera optical X right, Y down, Z forward.
"""
from __future__ import annotations
import numpy as np


def virtual_intrinsics(width: int = 640, height: int = 384, hfov_deg: float = 45.0) -> np.ndarray:
    if width <= 0 or height <= 0 or not 0 < hfov_deg < 180:
        raise ValueError('Invalid image dimensions or horizontal FOV')
    f = (width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
    return np.array([[f, 0., width / 2.], [0., f, height / 2.], [0., 0., 1.]], dtype=np.float64)


def remap_bgr(bgr, K_src, distortion=None, *, width=640, height=384,
              hfov_deg=45.0, border_value=(0, 0, 0)):
    """Inverse warp virtual optical rays into the original distorted camera.

    No extrinsic rotation is applied. Source distortion uses OpenCV order
    (k1,k2,p1,p2,k3, ...). Zero coefficients are valid for rectified images.
    """
    import cv2
    K_src = np.asarray(K_src, dtype=np.float64)
    if K_src.shape != (3, 3): raise ValueError('K_src must be 3x3')
    dist = np.zeros(5, dtype=np.float64) if distortion is None else np.asarray(distortion, np.float64).reshape(-1)
    K_dst = virtual_intrinsics(width, height, hfov_deg)
    mx, my = cv2.initUndistortRectifyMap(K_src, dist, np.eye(3), K_dst,
                                          (width, height), cv2.CV_32FC1)
    image = cv2.remap(bgr, mx, my, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
    return image, K_dst


def transform_camera_poses(world_ego, T_ego_camera):
    """Return first-camera-relative T_camera0_camerai, given T_world_ego.

    T_ego_camera maps camera optical coordinates into vehicle/ego coords.
    No fabricated virtual extrinsic: the same T_ego_camera is used throughout.
    """
    we = np.asarray(world_ego, dtype=np.float64)
    ec = np.asarray(T_ego_camera, dtype=np.float64)
    if we.ndim != 3 or we.shape[1:] != (4, 4):
        raise ValueError('world_ego must be (N,4,4)')
    if ec.shape == (4, 4): ec = np.broadcast_to(ec, we.shape)
    if ec.shape != we.shape: raise ValueError('T_ego_camera must be (4,4) or (N,4,4)')
    wc = we @ ec
    return np.linalg.inv(wc[0])[None] @ wc


def roundtrip_error(gt, reconstructed):
    from scipy.spatial.transform import Rotation
    gt, reconstructed = np.asarray(gt), np.asarray(reconstructed)
    err = np.linalg.inv(gt) @ reconstructed
    return (float(np.linalg.norm(err[:, :3, 3], axis=-1).max()),
            float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(err[:, :3, :3]).as_rotvec(), axis=-1)).max()))


def project_camera_xyz(xyz, K, *, min_z=0.5, max_z=50.0):
    xyz = np.asarray(xyz, np.float64)
    z = xyz[..., 2]
    valid = np.isfinite(xyz).all(axis=-1) & (z >= min_z) & (z <= max_z)
    safe_z = np.maximum(z, 1e-9)
    uv = np.stack([K[0, 0] * xyz[..., 0] / safe_z + K[0, 2],
                   K[1, 1] * xyz[..., 1] / safe_z + K[1, 2]], axis=-1)
    return uv, valid
