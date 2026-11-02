"""Shared pinhole virtual camera. Optical axis/center remain unchanged."""
from functools import lru_cache
import math
import cv2
import numpy as np

WIDTH = 640
HEIGHT = 384
HFOV_DEG = 45.0
FOCAL = WIDTH / (2.0 * math.tan(math.radians(HFOV_DEG) / 2.0))
K_VIRTUAL = np.array([[FOCAL, 0.0, WIDTH / 2.0],
                      [0.0, FOCAL, HEIGHT / 2.0],
                      [0.0, 0.0, 1.0]], dtype=np.float64)


def camera_metadata():
    return {"K_virtual": K_VIRTUAL.tolist(),
            "virtual_size": [WIDTH, HEIGHT], "virtual_hfov_deg": HFOV_DEG}


def validate_camera(meta):
    if not np.allclose(np.asarray(meta["K_virtual"], dtype=float), K_VIRTUAL, atol=1e-5):
        raise ValueError("Virtual intrinsic does not match this code. Rebuild data if settings changed.")
    if meta["virtual_size"] != [WIDTH, HEIGHT]:
        raise ValueError("Virtual image size does not match this code.")


@lru_cache(maxsize=12)
def _maps(w, h, intrinsics, distortion):
    k = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    d = np.asarray(distortion, dtype=np.float64)
    mx, my = cv2.initUndistortRectifyMap(
        k, d, np.eye(3), K_VIRTUAL, (WIDTH, HEIGHT), cv2.CV_32FC1)
    valid = ((mx >= 0) & (mx < w - 1) & (my >= 0) & (my < h - 1))
    return mx, my, float(valid.mean())


def virtualize_bgr(image, K_src, distortion, min_valid=0.95):
    """Remap source BGR into zero-distortion virtual pinhole BGR."""
    h, w = image.shape[:2]
    k_flat = tuple(np.asarray(K_src, dtype=np.float64).reshape(-1).tolist())
    d_flat = tuple(np.asarray(distortion, dtype=np.float64).reshape(-1).tolist())
    mx, my, coverage = _maps(w, h, k_flat, d_flat)
    if coverage < min_valid:
        raise ValueError(f"Virtual camera coverage {coverage:.1%} < {min_valid:.0%}; reduce FOV or inspect K/distortion")
    return cv2.remap(image, mx, my, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT), coverage
