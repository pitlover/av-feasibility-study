"""Geometrically consistent, rotation-only, common-intrinsic rectification.

All resulting action frames use OpenCV optical coordinates: +X image-right,
+Y image-down, +Z forward. Virtual optical center coincides with the physical
camera's optical center (no unsupported translation warping).
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import math
import cv2
import numpy as np

# Waymo camera is X-forward, Y-left, Z-up, per Waymo Perception documentation.
# T_waymoCamera_opencv: [x-forward,y-left,z-up] <- [x-right,y-down,z-forward]
R_WAYMO_CAMERA_FROM_OPTICAL = np.array([
    [0., 0., 1.],
    [-1., 0., 0.],
    [0., -1., 0.],
], dtype=np.float64)


@dataclass(frozen=True)
class VirtualSpec:
    width: int = 640
    height: int = 384
    hfov_deg: float = 45.0

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0 or self.width % 2 or self.height % 2:
            raise ValueError('width/height must be positive and even for H.264 yuv420p')
        if not 5 < self.hfov_deg < 170:
            raise ValueError('hfov_deg must be in (5,170)')

    @property
    def K(self):
        f = self.width / (2. * math.tan(math.radians(self.hfov_deg) / 2.))
        return np.array([[f, 0., self.width / 2.],
                         [0., f, self.height / 2.],
                         [0., 0., 1.]], dtype=np.float64)


def optical_from_source(dataset: str, extra_rotation: np.ndarray | None = None):
    """Return T_sourceSensor_virtualOptical, rotation-only, 4x4.

    K_src is already defined on OpenCV optical rays. For Waymo this fixes the
    non-OpenCV extrinsic sensor axes; for KITTI/AV2 source axes are optical.
    extra_rotation is optical-source <- virtual optical (camera orientation
    change), NOT a vehicle-camera extrinsic or an Euler-angle shortcut.
    """
    R = R_WAYMO_CAMERA_FROM_OPTICAL if dataset == 'waymo' else np.eye(3)
    if extra_rotation is not None:
        extra_rotation = np.asarray(extra_rotation, dtype=np.float64).reshape(3, 3)
        if not np.allclose(extra_rotation.T @ extra_rotation, np.eye(3), atol=1e-5) or np.linalg.det(extra_rotation) < 0.999:
            raise ValueError('extra rotation must be in SO(3)')
        R = R @ extra_rotation
    T = np.eye(4)
    T[:3, :3] = R
    return T


@lru_cache(maxsize=24)
def _cached_maps(K_src_bytes, dist_bytes, R_bytes, K_virtual_bytes, size_wh):
    K_src=np.frombuffer(K_src_bytes,dtype='<f8').reshape(3,3)
    d=np.frombuffer(dist_bytes,dtype='<f8')
    R=np.frombuffer(R_bytes,dtype='<f8').reshape(3,3)
    K_virtual=np.frombuffer(K_virtual_bytes,dtype='<f8').reshape(3,3)
    return cv2.initUndistortRectifyMap(K_src,d,R,K_virtual,size_wh,cv2.CV_32FC1)


def render_virtual(bgr: np.ndarray, K_src: np.ndarray, dist_cv: np.ndarray,
                   K_virtual: np.ndarray, size_wh: tuple[int,int],
                   R_source_optical_from_virtual: np.ndarray | None = None,
                   min_valid_fraction: float = 0.97):
    """Return remapped BGR frame plus valid pixel fraction.

    For an output virtual ray, OpenCV computes source ray R^-1 * ray_virtual;
    `R` below is source_optical -> virtual_optical. Thus R is transpose of
    `R_source_optical_from_virtual`.
    """
    if bgr is None or bgr.ndim != 3 or bgr.shape[-1] != 3:
        raise ValueError('Expected BGR HxWx3')
    K_src = np.asarray(K_src, dtype=np.float64).reshape(3, 3)
    K_virtual = np.asarray(K_virtual, dtype=np.float64).reshape(3, 3)
    d = np.asarray(dist_cv, dtype=np.float64).reshape(-1)
    if len(d) not in (4, 5, 8):
        raise ValueError(f'OpenCV pinhole distortion requires 4, 5 or 8 coefficients, got {len(d)}')
    R_src_virt = np.eye(3) if R_source_optical_from_virtual is None else np.asarray(R_source_optical_from_virtual, dtype=np.float64).reshape(3,3)
    R_src_to_virtual = R_src_virt.T
    mx, my = _cached_maps(np.asarray(K_src,dtype='<f8').tobytes(),
                           np.asarray(d,dtype='<f8').tobytes(),
                           np.asarray(R_src_to_virtual,dtype='<f8').tobytes(),
                           np.asarray(K_virtual,dtype='<f8').tobytes(), size_wh)
    h,w = bgr.shape[:2]
    valid = (np.isfinite(mx) & np.isfinite(my) &
             (mx >= 0) & (my >= 0) & (mx <= w-1) & (my <= h-1))
    valid_fraction = float(np.mean(valid))
    if valid_fraction < min_valid_fraction:
        raise ValueError(f'Virtual FOV exceeds source valid pixels: {valid_fraction:.3%} < {min_valid_fraction:.3%}; reduce --hfov or inspect calibration')
    out = cv2.remap(bgr, mx, my, interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return out, valid_fraction
