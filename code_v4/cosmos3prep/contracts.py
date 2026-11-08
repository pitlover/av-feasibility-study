"""Safe V2/V4 camera-relative GT metadata contract resolution.

No dataset files are modified by this module.  In V2, the camera-local
GT actions were stored without an `action_frame` key, and the declared pose
file was `poses_local_virtual_camera.npy`.  Do not interpret those actions
as ego-frame motion.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

V2 = 'cosmos3_virtual_camera_v2'
CAMERA_POSES = ('poses_local_virtual_camera.npy', 'poses_local_camera.npy')
EGO_POSES = ('poses_local_ego.npy',)


def action_frame(meta: dict) -> str:
    """Resolve action coordinates, using explicit schema for V2.

    Deliberately avoids guessing from numeric trajectory appearance. V2
    virtual camera documents camera-frame 9D actions even when the legacy
    action_frame key is absent. Other old datasets default to ego.
    """
    fmt = meta.get('format')
    raw = meta.get('action_frame')
    if fmt == V2:
        if raw not in (None, '', 'camera'):
            raise ValueError('V2 virtual-camera actions must be camera-frame, not ego')
        name = meta.get('pose_local_file')
        if name not in CAMERA_POSES:
            raise ValueError(f'V2 metadata has unrecognized camera pose_local_file: {name!r}')
        return 'camera'
    if raw in ('ego', 'camera'):
        return raw
    if raw is not None:
        raise ValueError(f'Unsupported action_frame: {raw!r}')
    return 'ego'


def action_pose_path(meta: dict, folder: str | Path) -> Path:
    """Select the pose file matching the GT actions; never silently fall to ego."""
    frame = action_frame(meta)
    folder = Path(folder)
    if meta.get('format') == V2:
        # Explicit V2 file takes precedence. Some server revisions also wrote
        # poses_local_camera.npy; use it only when the declared file is absent.
        names = (meta['pose_local_file'], *CAMERA_POSES)
    else:
        canonical = 'poses_local_camera.npy' if frame == 'camera' else 'poses_local_ego.npy'
        declared = meta.get('pose_local_file')
        if declared is not None and declared != canonical:
            raise ValueError(f'Pose metadata conflicts with action_frame={frame}: {declared!r}')
        names = (canonical,)
    for name in dict.fromkeys(names):
        if name not in CAMERA_POSES + EGO_POSES:
            raise ValueError(f'Unsupported pose filename: {name!r}')
        p = folder / name
        if p.is_file():
            return p
    raise FileNotFoundError(f'No {frame}-frame pose file in {folder}; expected {names}')


def image_mode(meta: dict) -> str:
    mode = meta.get('image_mode')
    if meta.get('format') == V2:
        if mode not in (None, 'virtual'):
            raise ValueError(f'Invalid V2 virtual-camera image_mode: {mode!r}')
        return 'virtual'
    return mode or 'letterbox'


def video_intrinsics(meta: dict) -> np.ndarray:
    """Get the saved calibration, without inventing a virtual-camera FOV.

    Supports common v2/v4 naming. For a V2 file with no K, callers should
    inspect its complete metadata rather than silently assume 45 degrees.
    """
    possibilities = [
        meta.get('K_video'),
        meta.get('K_virtual'),
        meta.get('virtual_K'),
    ]
    camera = meta.get('camera')
    if isinstance(camera, dict):
        possibilities.extend([camera.get('K_virtual'), camera.get('K_video')])
    virtual_camera = meta.get('virtual_camera')
    if isinstance(virtual_camera, dict):
        possibilities.extend([virtual_camera.get('K_virtual'), virtual_camera.get('K')])
    for item in possibilities:
        if item is None:
            continue
        k = np.asarray(item, dtype=np.float64)
        if k.shape == (3, 3) and np.isfinite(k).all() and k[0,0] > 0 and k[1,1] > 0:
            return k
    raise ValueError('No valid virtual-video intrinsic K in metadata (K_video/K_virtual).')


def normalized_metadata(meta: dict) -> dict:
    """In-memory metadata view for downstream plotting; does not write files."""
    result = dict(meta)
    result['action_frame'] = action_frame(meta)
    result['image_mode'] = image_mode(meta)
    if result['image_mode'] == 'virtual':
        try:
            result['K_video'] = video_intrinsics(meta).tolist()
        except ValueError:
            # Numeric pose evaluation does not need K. Rendering will explain
            # the missing calibration if the user requests a camera overlay.
            pass
    return result
