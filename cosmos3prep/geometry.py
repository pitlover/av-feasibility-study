"""SE(3), pose interpolation and Cosmos-compatible 9D relative action encoding."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def matrix4(x, *, shape=(4, 4)) -> np.ndarray:
    """Read a row-major 3x4 or 4x4 rigid transform into float64 4x4."""
    a = np.asarray(x, dtype=np.float64)
    if a.size == 12:
        a = a.reshape(3, 4)
        t = np.eye(4, dtype=np.float64)
        t[:3] = a
    elif a.size == 16:
        t = a.reshape(4, 4).copy()
    else:
        raise ValueError(f"Transform must have 12/16 elements, got {a.size}")
    assert t.shape == shape
    return t


def from_wxyz_row(row) -> np.ndarray:
    """Argoverse Feather row with quaternion wxyz and translation in meters."""
    q = np.array([row['qx'], row['qy'], row['qz'], row['qw']], dtype=np.float64)
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = Rotation.from_quat(q / np.linalg.norm(q)).as_matrix()
    t[:3, 3] = [row['tx_m'], row['ty_m'], row['tz_m']]
    return t


def interpolate_pose(times_ns: np.ndarray, poses: np.ndarray, query_ns: int,
                     max_gap_ns: int = 120_000_000) -> np.ndarray | None:
    """Exact pose or linear translation + SLERP; never extrapolate or cross large gaps."""
    j = int(np.searchsorted(times_ns, query_ns, side='left'))
    if j < len(times_ns) and int(times_ns[j]) == int(query_ns):
        return poses[j].copy()
    if j == 0 or j == len(times_ns):
        return None
    left, right = int(times_ns[j - 1]), int(times_ns[j])
    if right <= left or right - left > max_gap_ns:
        return None
    f = (int(query_ns) - left) / (right - left)
    ret = np.eye(4, dtype=np.float64)
    ret[:3, 3] = (1 - f) * poses[j - 1, :3, 3] + f * poses[j, :3, 3]
    keyrots = Rotation.from_matrix(poses[[j - 1, j], :3, :3])
    ret[:3, :3] = Slerp([0., 1.], keyrots)([f]).as_matrix()[0]
    return ret


def localize(poses: np.ndarray) -> np.ndarray:
    """Convert world<-ego trajectory to first-frame-relative poses; meters preserved."""
    p = np.asarray(poses, np.float64)
    return np.linalg.inv(p[0])[None] @ p


def _rot6_encode(r: np.ndarray) -> np.ndarray:
    # NVIDIA uses column-based: [R[:,0], R[:,1]], not row-major [:,:2].reshape(-1)
    return np.concatenate([r[..., :, 0], r[..., :, 1]], axis=-1)


def _rot6_decode(r6: np.ndarray) -> np.ndarray:
    a = r6[..., 0:3]
    b = r6[..., 3:6]
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    b = b - (a * b).sum(axis=-1, keepdims=True) * a
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
    c = np.cross(a, b)
    return np.stack([a, b, c], axis=-1)


def encode_local(poses: np.ndarray, convention: str) -> np.ndarray:
    """Pure-numpy equivalent of Cosmos pose_abs_to_rel(..., rot6d)."""
    if convention == 'backward_framewise':
        delta = np.linalg.inv(poses[:-1]) @ poses[1:]
    elif convention == 'backward_anchored':
        delta = np.linalg.inv(poses[0])[None] @ poses[1:]
    else:
        raise ValueError(f"Unsupported convention: {convention}")
    return np.concatenate([delta[:, :3, 3], _rot6_encode(delta[:, :3, :3])], -1).astype(np.float32)


def decode_local(actions: np.ndarray, convention: str,
                 initial_pose: np.ndarray | None = None) -> np.ndarray:
    a = np.asarray(actions, np.float64)
    if a.ndim != 2 or a.shape[1] != 9:
        raise ValueError(f"Expected actions (N,9), got {a.shape}")
    p0 = np.eye(4, dtype=np.float64) if initial_pose is None else np.asarray(initial_pose, np.float64)
    out = [p0.copy()]
    for row in a:
        delta = np.eye(4, dtype=np.float64)
        delta[:3, 3] = row[:3]
        delta[:3, :3] = _rot6_decode(row[3:])
        if convention == 'backward_framewise':
            out.append(out[-1] @ delta)
        elif convention == 'backward_anchored':
            out.append(p0 @ delta)
        else:
            raise ValueError(convention)
    return np.stack(out)


def encode_actions(poses: np.ndarray, convention: str, encoder: str = 'auto') -> tuple[np.ndarray, str]:
    """Prefer NVIDIA's official function if installed; strict 'official' option."""
    if encoder not in ('auto', 'official', 'local'):
        raise ValueError(encoder)
    if encoder != 'local':
        try:
            from cosmos_framework.data.generator.action.utils.pose_utils import pose_abs_to_rel
        except ImportError:
            if encoder == 'official':
                raise RuntimeError('Cosmos Framework not importable. Install it, or use --encoder local.')
        else:
            action = pose_abs_to_rel(
                np.asarray(poses, dtype=np.float32),
                rotation_format='rot6d',
                pose_convention=convention,
            )
            return np.asarray(action, dtype=np.float32), 'official_cosmos_framework'
    return encode_local(poses, convention), 'local_cosmos_compatible'


def validate_pose_array(poses: np.ndarray) -> None:
    if not np.isfinite(poses).all():
        raise ValueError('Nonfinite GT pose')
    if not np.allclose(poses[:, 3], [0., 0., 0., 1.], atol=1e-5):
        raise ValueError('Invalid homogeneous transform bottom row')
    r = poses[:, :3, :3]
    orth = np.max(np.abs(np.swapaxes(r, -1, -2) @ r - np.eye(3)))
    det = np.max(np.abs(np.linalg.det(r) - 1.0))
    if orth > 1e-3 or det > 1e-3:
        raise ValueError(f'Invalid SO(3) rotation: orth={orth}, det={det}')
