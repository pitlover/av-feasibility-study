"""Memory-aware nuScenes v1.0-trainval CAM_FRONT adapter for the Cosmos3 ablation pipeline.

Reads official JSON metadata only, no nuscenes-devkit dependency. Uses BOTH
samples/ (2Hz key frames) and sweeps/ (~12Hz intermediate frames), following
sample_data.next within each scene. Ground truth uses each camera frame's own
ego_pose and calibrated_sensor: T_global_camera = T_global_ego @ T_ego_camera.

Canonical nuScenes train/val split is taken from the nuScenes devkit
`nuscenes.utils.splits.val` (150 scene names). All other scenes in the
v1.0-trainval metadata are train. This is NOT a random split.

Images in nuScenes camera records are provided undistorted/rectified; the
camera_intrinsic belongs to that image. No additional distortion correction
is applied when preserving original images.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
from scipy.spatial.transform import Rotation

from .core import Frame

# nuScenes devkit train/val split, from nutonomy/nuscenes-devkit,
# python-sdk/nuscenes/utils/splits.py (`val` constant).
VAL_SCENES = frozenset("""
scene-0003 scene-0012 scene-0013 scene-0014 scene-0015 scene-0016 scene-0017 scene-0018
scene-0035 scene-0036 scene-0038 scene-0039 scene-0092 scene-0093 scene-0094 scene-0095
scene-0096 scene-0097 scene-0098 scene-0099 scene-0100 scene-0101 scene-0102 scene-0103
scene-0104 scene-0105 scene-0106 scene-0107 scene-0108 scene-0109 scene-0110 scene-0221
scene-0268 scene-0269 scene-0270 scene-0271 scene-0272 scene-0273 scene-0274 scene-0275
scene-0276 scene-0277 scene-0278 scene-0329 scene-0330 scene-0331 scene-0332 scene-0344
scene-0345 scene-0346 scene-0519 scene-0520 scene-0521 scene-0522 scene-0523 scene-0524
scene-0552 scene-0553 scene-0554 scene-0555 scene-0556 scene-0557 scene-0558 scene-0559
scene-0560 scene-0561 scene-0562 scene-0563 scene-0564 scene-0565 scene-0625 scene-0626
scene-0627 scene-0629 scene-0630 scene-0632 scene-0633 scene-0634 scene-0635 scene-0636
scene-0637 scene-0638 scene-0770 scene-0771 scene-0775 scene-0777 scene-0778 scene-0780
scene-0781 scene-0782 scene-0783 scene-0784 scene-0794 scene-0795 scene-0796 scene-0797
scene-0798 scene-0799 scene-0800 scene-0802 scene-0904 scene-0905 scene-0906 scene-0907
scene-0908 scene-0909 scene-0910 scene-0911 scene-0912 scene-0913 scene-0914 scene-0915
scene-0916 scene-0917 scene-0919 scene-0920 scene-0921 scene-0922 scene-0923 scene-0924
scene-0925 scene-0926 scene-0927 scene-0928 scene-0929 scene-0930 scene-0931 scene-0962
scene-0963 scene-0966 scene-0967 scene-0968 scene-0969 scene-0971 scene-0972 scene-1059
scene-1060 scene-1061 scene-1062 scene-1063 scene-1064 scene-1065 scene-1066 scene-1067
scene-1068 scene-1069 scene-1070 scene-1071 scene-1072 scene-1073
""".split())

if len(VAL_SCENES) != 150:
    raise RuntimeError(f"Expected 150 official nuScenes validation scenes, got {len(VAL_SCENES)}")


def stream_json_array(path: Path) -> Iterator[dict]:
    """Stream a giant top-level JSON array using stdlib, without json.load OOM.

    sample_data.json contains millions of sensor records; we keep only the
    selected camera's records. The parser handles compact/pretty-printed JSON.
    """
    decoder = json.JSONDecoder()
    with Path(path).open("r", encoding="utf-8") as fh:
        buf = ""
        pos = 0
        eof = False
        began = False
        while True:
            if pos >= len(buf) or len(buf) - pos < 128:
                buf = buf[pos:] + fh.read(1 << 20)
                pos = 0
            if not buf:
                raise ValueError(f"Unexpected EOF in JSON array: {path}")
            while pos < len(buf) and buf[pos] in " \t\r\n,":
                pos += 1
            if pos >= len(buf):
                continue
            if not began:
                if buf[pos] != "[":
                    raise ValueError(f"Expected a top-level JSON array in {path}")
                began = True
                pos += 1
                continue
            if buf[pos] == "]":
                if fh.read(1).strip():
                    raise ValueError(f"Unexpected trailing content in {path}")
                return
            try:
                obj, next_pos = decoder.raw_decode(buf, pos)
            except json.JSONDecodeError:
                # Incomplete record straddling a chunk; keep unread suffix.
                extra = fh.read(1 << 20)
                if not extra:
                    raise ValueError(f"Incomplete JSON record at offset in {path}")
                buf = buf[pos:] + extra
                pos = 0
                continue
            if not isinstance(obj, dict):
                raise ValueError(f"JSON array item is not an object: {path}")
            yield obj
            pos = next_pos


def make_transform(translation: list, quaternion_wxyz: list) -> np.ndarray:
    """nuScenes quaternion is [w,x,y,z], transform maps local -> parent."""
    qw, qx, qy, qz = map(float, quaternion_wxyz)
    xyz = np.asarray(translation, dtype=np.float64).reshape(3)
    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    if not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-12:
        raise ValueError("Bad nuScenes rotation quaternion")
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_quat(quat / np.linalg.norm(quat)).as_matrix()
    T[:3, 3] = xyz
    return T


def _read_metadata(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"nuScenes metadata missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def iter_nuscenes(root: Path, camera: str = "CAM_FRONT", max_logs: int = 0,
                  *, strict: bool = False):
    """Yield `(scene_name, train|val, list[Frame])` in scene order.

    Only the camera's sample_data metadata is retained in memory. All 12Hz
    sweeps are accessed by following the first keyframe -> last keyframe.
    We stop at the LAST keyframe to avoid crossing scene boundaries.
    """
    root = Path(root).expanduser().resolve()
    version = root / "v1.0-trainval"
    if not version.is_dir():
        raise FileNotFoundError(f"Expected nuScenes metadata folder: {version}")
    for prefix in ("samples", "sweeps"):
        if not (root / prefix).is_dir():
            raise FileNotFoundError(f"nuScenes directory missing: {root / prefix}")
    scenes = _read_metadata(version / "scene.json")
    samples = {r["token"]: r for r in _read_metadata(version / "sample.json")}
    sensors = {r["token"]: r for r in _read_metadata(version / "sensor.json")}
    selected_sensor_ids = {tok for tok, r in sensors.items() if r.get("channel") == camera and r.get("modality") == "camera"}
    if not selected_sensor_ids:
        raise ValueError(f"Unknown nuScenes camera {camera!r}; check sensor.json")
    calibrations = {r["token"]: r for r in _read_metadata(version / "calibrated_sensor.json")
                    if r["sensor_token"] in selected_sensor_ids}
    if not calibrations:
        raise ValueError(f"No calibrated_sensor entries for {camera}")
    # ego_pose is about 2.6M records in a full trainval; filter by camera
    # ego_pose tokens in the same pass rather than loading the entire JSON.
    camera_data = {}
    keyframe_for_sample = {}
    camera_cal_tokens = set(calibrations)
    inspected = 0
    for sd in stream_json_array(version / "sample_data.json"):
        inspected += 1
        if sd.get("calibrated_sensor_token") not in camera_cal_tokens:
            continue
        camera_data[sd["token"]] = sd
        if sd.get("is_key_frame"):
            sample_token = sd.get("sample_token")
            if sample_token:
                if sample_token in keyframe_for_sample:
                    raise ValueError(f"Duplicate keyframe for {camera}/{sample_token}")
                keyframe_for_sample[sample_token] = sd["token"]
    print(f"[nuScenes metadata] sample_data={inspected:,}, {camera} frames={len(camera_data):,}, "
          f"keyframes={len(keyframe_for_sample):,}")
    if not camera_data:
        raise ValueError(f"No {camera} sample_data records")
    # Restrict full ego_pose read to camera-referenced pose entries.
    needed_pose_tokens = {sd["ego_pose_token"] for sd in camera_data.values()}
    camera_poses = {}
    for pose in stream_json_array(version / "ego_pose.json"):
        if pose["token"] in needed_pose_tokens:
            camera_poses[pose["token"]] = pose
    if len(camera_poses) != len(needed_pose_tokens):
        raise ValueError(f"Missing {len(needed_pose_tokens)-len(camera_poses)} camera ego poses")
    print(f"[nuScenes metadata] camera ego_pose={len(camera_poses):,} loaded; "
          f"official val scenes={len(VAL_SCENES)}")
    selected = sorted(scenes, key=lambda s: s["name"])
    if max_logs:
        selected = selected[:max_logs]
    for scene in selected:
        scene_name = scene["name"]
        split = "val" if scene_name in VAL_SCENES else "train"
        if not scene.get("first_sample_token") or not scene.get("last_sample_token"):
            raise ValueError(f"Incomplete scene metadata for {scene_name}")
        for endpoint in ("first_sample_token", "last_sample_token"):
            sample = samples.get(scene[endpoint])
            if sample is None or sample.get("scene_token") != scene["token"]:
                raise ValueError(f"{scene_name}: {endpoint} does not belong to this scene")
        first = keyframe_for_sample.get(scene["first_sample_token"])
        last = keyframe_for_sample.get(scene["last_sample_token"])
        if first is None or last is None:
            msg = f"{scene_name}: missing {camera} first/last keyframe sample_data"
            if strict:
                raise ValueError(msg)
            print("[skip]", msg)
            continue
        frames: list[Frame] = []
        token = first
        visited = set()
        while token:
            if token in visited:
                raise ValueError(f"{scene_name}: camera sample_data cycle at {token}")
            visited.add(token)
            sd = camera_data.get(token)
            if sd is None:
                msg = f"{scene_name}: missing {camera} record in linked chain {token}"
                if strict:
                    raise ValueError(msg)
                print("[warn]", msg)
                break
            calib = calibrations.get(sd["calibrated_sensor_token"])
            pose = camera_poses.get(sd["ego_pose_token"])
            if calib is None or pose is None:
                raise ValueError(f"{scene_name}: missing calibration/pose for {token}")
            filename = sd.get("filename", "")
            path = root / filename
            if not path.is_file():
                msg = f"{scene_name}: missing image {path}"
                if strict:
                    raise FileNotFoundError(msg)
                print("[warn]", msg)
                # Missing-image breaks continuity; do not bridge the gap.
            else:
                K = np.asarray(calib["camera_intrinsic"], dtype=np.float64)
                if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
                    raise ValueError(f"Invalid nuScenes intrinsic for {scene_name}/{token}")
                frame = Frame(
                    t_s=float(sd["timestamp"]) / 1e6,
                    image=path,
                    world_ego=make_transform(pose["translation"], pose["rotation"]),
                    ego_camera=make_transform(calib["translation"], calib["rotation"]),
                    K=K,
                    distortion=np.zeros(5, dtype=np.float64),
                    needs_undistort=False,
                    source_id=f"{scene_name}/{camera}/{token}",
                )
                frames.append(frame)
            if token == last:
                break
            token = sd.get("next", "")
        if token != last:
            msg = f"{scene_name}: linked chain ended before last camera keyframe"
            if strict:
                raise ValueError(msg)
            print("[warn]", msg)
        yield scene_name, split, frames
