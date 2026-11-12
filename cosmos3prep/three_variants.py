"""Camera/FPS ablation preprocessors for Cosmos3 AV inverse dynamics.

Install alongside the *existing* cosmos3prep package on feat/prepare_nuscenes.
This module deliberately does not modify the original v2 pipeline, and writes
independent output roots so prior processed clips cannot be overwritten.

Experiments
-----------
original_10hz    Native pixels/K, sampled at 10 Hz.
original_native  Native pixels/K, original camera cadence (KITTI 10, Waymo 10, AV2 20).
virtual_native   Common virtual pinhole K (640x384 / 45 deg), original cadence.

nuScenes supports original_native and virtual_native, both with its native ~12Hz CAM_FRONT stream.

The camera extrinsics are always the physical sensor extrinsics.  All variants
produce 61 RGB frames, 60 camera-relative 9D actions, and real timestamps.
The 20 Hz variants cover ~3 s rather than ~6 s at 10 Hz.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .core import Frame, frame_runs_10hz, frame_runs_kitti_native, rebuild_index, safe_id
from .datasets import iter_av2, iter_kitti, iter_waymo
from .nuscenes_adapter import iter_nuscenes
from .geometry import decode_local, encode_actions, localize, validate_pose_array
from .contracts import action_pose_path
from cosmos3virtual.camera import remap_bgr, virtual_intrinsics

VARIANTS = ("original_10hz", "original_native", "virtual_native")
N_FRAMES = 61
N_ACTIONS = 60
NOMINAL_SOURCE_FPS = {"kitti": 10, "waymo": 10, "av2": 20, "nuscenes": 12}
DEFAULT_CAMERA = {"kitti": "image_2", "waymo": "FRONT", "av2": "stereo_front_left", "nuscenes": "CAM_FRONT"}


def target_fps(dataset: str, variant: str) -> int:
    if dataset not in NOMINAL_SOURCE_FPS or variant not in VARIANTS:
        raise ValueError(f"Invalid dataset/variant: {dataset}/{variant}")
    if dataset == "nuscenes" and variant == "original_10hz":
        raise ValueError("nuScenes supports two 12Hz variants only: original_native, virtual_native")
    return 10 if variant == "original_10hz" else NOMINAL_SOURCE_FPS[dataset]


def is_virtual(variant: str) -> bool:
    return variant == "virtual_native"


def native_runs(frames: list[Frame], dataset: str, fps: int) -> list[list[Frame]]:
    """Retain original images and timestamps; split on missing camera frames.

    Uses a moderate *inter-frame* tolerance, not an artificial global grid.
    Does not duplicate RGB, interpolate RGB, or fill dropped frames.
    """
    if dataset == "kitti":
        # Existing KITTI-specific path checks BOTH timestamps and image indices.
        return frame_runs_kitti_native(frames)
    seq = sorted(frames, key=lambda f: f.t_s)
    if not seq:
        return []
    ideal = 1.0 / fps
    # Accept modest clock jitter but reject missing 20 Hz images (gap ~0.1 s).
    tolerance = {10: 0.04, 12: 0.04, 20: 0.0225}.get(fps, 0.025)
    runs: list[list[Frame]] = []
    current = [seq[0]]
    for prev, cur in zip(seq, seq[1:]):
        dt = cur.t_s - prev.t_s
        if math.isfinite(dt) and 0 < dt and abs(dt - ideal) <= tolerance:
            current.append(cur)
        else:
            runs.append(current)
            current = [cur]
    runs.append(current)
    return runs


def select_runs(frames: list[Frame], dataset: str, variant: str,
                tolerance_ms: float = 35.0) -> list[list[Frame]]:
    fps = target_fps(dataset, variant)
    if variant == "original_10hz":
        if dataset == "kitti":
            # KITTI nominal 10 Hz acquisition is not anchored to a global grid.
            return frame_runs_kitti_native(frames)
        return frame_runs_10hz(frames, fps=10, tolerance_s=tolerance_ms / 1000.0)
    return native_runs(frames, dataset=dataset, fps=fps)


def _raw_image(frame: Frame) -> np.ndarray:
    """Decode original camera pixels WITHOUT undistorting Waymo twice."""
    if isinstance(frame.image, bytes):
        result = cv2.imdecode(np.frombuffer(frame.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    else:
        result = cv2.imread(str(frame.image), cv2.IMREAD_COLOR)
    if result is None:
        raise ValueError(f"Image decode failed: {frame.source_id}")
    return result


def _camera_frame(frame: Frame, variant: str, width: int, height: int,
                  hfov: float) -> np.ndarray:
    original = _raw_image(frame)
    if is_virtual(variant):
        processed, _ = remap_bgr(original, frame.K, frame.distortion,
                                 width=width, height=height, hfov_deg=hfov)
        return processed
    # 'original' deliberately preserves the source's pixel geometry.  In
    # particular, Waymo lens distortion is NOT silently removed here.
    return original


def _video_geometry(first: Frame, variant: str, width: int, height: int,
                    hfov: float) -> tuple[tuple[int, int], tuple[int, int], np.ndarray]:
    original = _raw_image(first)
    orig_h, orig_w = original.shape[:2]
    if is_virtual(variant):
        if width % 2 or height % 2:
            raise ValueError("Virtual output dimensions must be even (H.264 yuv420p)")
        return (orig_w, orig_h), (width, height), virtual_intrinsics(width, height, hfov)
    # H.264/yuv420p needs even dimensions. Pad on RIGHT/BOTTOM only; the original
    # pixels and their K remain unchanged. Report the actual encoded canvas.
    even_w = orig_w + (orig_w % 2)
    even_h = orig_h + (orig_h % 2)
    return (orig_w, orig_h), (even_w, even_h), np.asarray(first.K, np.float64).copy()


def _write_video(frames: list[Frame], dst: Path, fps: int, *, variant: str,
                 width: int, height: int, hfov: float, crf: int, preset: str) -> tuple[tuple[int, int], tuple[int, int], np.ndarray]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg + libx264 required to write H.264 MP4")
    source_wh, output_wh, K_video = _video_geometry(frames[0], variant, width, height, hfov)
    out_w, out_h = output_wh
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s:v", f"{out_w}x{out_h}",
           "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
           "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(dst)]
    with tempfile.TemporaryFile(mode="w+b") as err:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=err)
        try:
            assert proc.stdin is not None
            for frame in frames:
                bgr = _camera_frame(frame, variant, width, height, hfov)
                h, w = bgr.shape[:2]
                required_wh = (width, height) if is_virtual(variant) else source_wh
                if (w, h) != required_wh:
                    raise ValueError(f"Camera image size changed within clip: {(w, h)} != {required_wh}")
                if (w, h) != output_wh:
                    bgr = cv2.copyMakeBorder(bgr, 0, out_h - h, 0, out_w - w,
                                             cv2.BORDER_REPLICATE)
                proc.stdin.write(np.ascontiguousarray(bgr).tobytes())
            proc.stdin.close()
            result = proc.wait()
        except Exception as exc:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            err.seek(0)
            detail = err.read().decode(errors="replace")
            if isinstance(exc, BrokenPipeError):
                raise RuntimeError(f"ffmpeg broken pipe: {detail}") from exc
            raise
        err.seek(0)
        detail = err.read().decode(errors="replace")
    if result != 0 or not dst.is_file() or dst.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg exited {result}: {detail}")
    return source_wh, output_wh, K_video


def _save_json(path: Path, item: dict) -> None:
    path.write_text(json.dumps(item, indent=2, ensure_ascii=False), encoding="utf-8")


def _as_distortion_matrix(frames: list[Frame]) -> np.ndarray:
    raw = [np.asarray(f.distortion, np.float64).ravel() for f in frames]
    size = max(map(len, raw))
    result = np.zeros((len(raw), size), np.float64)
    for i, v in enumerate(raw):
        result[i, :len(v)] = v
    return result


def write_variant_clip(frames: list[Frame], *, dataset: str, sequence: str,
                       split: str, variant: str, first_index: int, out_root: Path,
                       encoder: str = "auto", crf: int = 18, preset: str = "veryfast",
                       width: int = 640, height: int = 384, hfov: float = 45.0,
                       camera: str | None = None, overwrite: bool = False) -> dict | None:
    """Write one complete 61-frame clip; atomic cleanup if any part fails."""
    if len(frames) != N_FRAMES:
        raise ValueError(f"Expected 61 frames, got {len(frames)}")
    fps = target_fps(dataset, variant)
    dst_id = f"{safe_id(dataset)}_{safe_id(sequence)}_{first_index:06d}"
    dst = Path(out_root) / variant / dataset / split / dst_id
    if dst.exists():
        if overwrite:
            shutil.rmtree(dst)
        elif (dst / "metadata.json").is_file() and (dst / "video.mp4").is_file():
            print(f"  [exists] {dst_id}")
            return None
        else:
            raise RuntimeError(f"Incomplete output folder: {dst}; use --overwrite")
    dst.mkdir(parents=True, exist_ok=False)
    try:
        ts = np.asarray([f.t_s for f in frames], np.float64)
        if not np.all(np.isfinite(ts)) or not np.all(np.diff(ts) > 0):
            raise ValueError("Nonfinite or nonincreasing capture timestamps")
        poses_world_ego = np.stack([f.world_ego for f in frames]).astype(np.float64)
        T_ego_camera = np.stack([f.ego_camera for f in frames]).astype(np.float64)
        validate_pose_array(poses_world_ego)
        validate_pose_array(T_ego_camera)
        poses_local_ego = localize(poses_world_ego)
        poses_local_camera = localize(poses_world_ego @ T_ego_camera)
        actions, encoder_used = encode_actions(poses_local_camera, "backward_framewise", encoder)
        if actions.shape != (N_ACTIONS, 9) or not np.isfinite(actions).all():
            raise ValueError(f"Incorrect action shape / values: {actions.shape}")
        recon = decode_local(actions, "backward_framewise")
        translation_err = float(np.max(np.linalg.norm(
            recon[:, :3, 3] - poses_local_camera[:, :3, 3], axis=1)))
        rel_R = np.swapaxes(recon[:, :3, :3], 1, 2) @ poses_local_camera[:, :3, :3]
        cos = np.clip((np.trace(rel_R, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
        rotation_err = float(np.max(np.degrees(np.arccos(cos))))
        if translation_err > 0.002 or rotation_err > 0.05:
            raise ValueError(f"GT roundtrip mismatch: {translation_err:.6f} m, {rotation_err:.6f} deg")

        source_wh, output_wh, K_first = _write_video(
            frames, dst / "video.mp4", fps, variant=variant,
            width=width, height=height, hfov=hfov, crf=crf, preset=preset)
        K_source_all = np.stack([np.asarray(f.K, np.float64) for f in frames])
        K_video_all = (np.broadcast_to(K_first, (N_FRAMES, 3, 3)).copy()
                       if is_virtual(variant) else K_source_all.copy())
        distortion_source_all = _as_distortion_matrix(frames)
        stored = {
            "timestamps_s.npy": ts,
            "poses_world_ego.npy": poses_world_ego,
            "poses_local_ego.npy": poses_local_ego,
            "poses_local_camera.npy": poses_local_camera,
            "T_ego_camera.npy": T_ego_camera,
            "actions_raw_9d.npy": actions,
            "K_source_per_frame.npy": K_source_all,
            "K_video_per_frame.npy": K_video_all,
            "distortion_source_per_frame.npy": distortion_source_all,
        }
        for filename, data in stored.items():
            np.save(dst / filename, data)
        dt = np.diff(ts)
        original_mode = not is_virtual(variant)
        video_distortion_corrected = (not original_mode) or dataset in ("kitti", "av2", "nuscenes")
        meta = {
            "format": "cosmos3_av_preprocess_v4",
            "ablation_variant": variant,
            "dataset": dataset, "sequence": sequence, "split": split,
            "clip_id": dst_id, "camera": camera or DEFAULT_CAMERA[dataset],
            "fps": fps, "source_nominal_fps": NOMINAL_SOURCE_FPS[dataset],
            "frame_count": N_FRAMES, "action_count": N_ACTIONS,
            "video_codec": "h264", "video_pixel_format": "yuv420p",
            "video_size_wh": list(output_wh), "source_image_size_wh": list(source_wh),
            "image_mode": "virtual" if is_virtual(variant) else "original",
            "image_processing": (
                "virtual_pinhole_undistort_reproject" if is_virtual(variant)
                else "original_pixels_no_undistort_right_bottom_even_padding_if_needed"
            ),
            "virtual_hfov_deg": hfov if is_virtual(variant) else None,
            "virtual_extrinsic_policy": "intrinsic_only_preserve_T_ego_camera",
            "action_frame": "camera", "camera_coordinate_axes": "x_right_y_down_z_forward",
            "pose_frame": {"kitti": "velodyne", "waymo": "vehicle", "av2": "egovehicle",
                           "nuscenes": "egovehicle"}[dataset],
            "pose_transform_semantics": "T_target_source, homogeneous column vector, meters",
            "pose_world_file": "poses_world_ego.npy",
            "pose_local_file": "poses_local_camera.npy",
            "pose_camera_local_file": "poses_local_camera.npy",
            "action_file": "actions_raw_9d.npy",
            "action_encoding": "translation3 + rot6d_column_based",
            "action_normalization": "none", "pose_convention": "backward_framewise",
            "action_encoder": encoder_used,
            "first_timestamp_s": float(ts[0]), "last_timestamp_s": float(ts[-1]),
            "actual_capture_duration_s": float(ts[-1] - ts[0]),
            "actual_effective_fps": float(N_ACTIONS / (ts[-1] - ts[0])),
            "clip_timebase": "source_native" if variant != "original_10hz" else "nominal_10hz",
            "delta_t_min_s": float(dt.min()), "delta_t_max_s": float(dt.max()),
            "video_path": str((dst / "video.mp4").resolve()),
            "K_source": K_source_all[0].tolist(), "K_video": K_video_all[0].tolist(),
            "K_source_per_frame_file": "K_source_per_frame.npy",
            "K_video_per_frame_file": "K_video_per_frame.npy",
            "distortion_source": distortion_source_all[0].tolist(),
            "distortion_source_per_frame_file": "distortion_source_per_frame.npy",
            "video_distortion_corrected": bool(video_distortion_corrected),
            "source_pixel_geometry": (
                "rectified" if dataset in ("kitti", "av2", "nuscenes") else "raw_distorted"
            ),
            "T_ego_camera_first": T_ego_camera[0].tolist(),
            "GT_roundtrip_max_translation_m": translation_err,
            "GT_roundtrip_max_rotation_deg": rotation_err,
            "source_frame_ids": [f.source_id for f in frames],
        }
        _save_json(dst / "metadata.json", meta)
        _save_json(dst / "cosmos_inference.json", {
            "action_chunk_size": N_ACTIONS, "domain_name": "av", "fps": fps,
            "image_size": 480, "model_mode": "inverse_dynamics",
            "name": dst_id, "prompt": "You are an autonomous vehicle planning system.",
            "seed": 0, "view_point": "ego_view",
            "vision_path": str((dst / "video.mp4").resolve()),
        })
        return {
            "clip_id": dst_id, "dataset": dataset, "sequence": sequence,
            "split": split, "video": str((dst / "video.mp4").resolve()),
            "meta": str((dst / "metadata.json").resolve()),
            "action_file": str((dst / "actions_raw_9d.npy").resolve()),
        }
    except Exception:
        shutil.rmtree(dst, ignore_errors=True)
        raise


def validate_clips(out: Path, max_clips: int = 0, verify_video: bool = True,
                   only_dataset: str | None = None) -> dict:
    """Validate native- and virtual-FPS variants without assuming 10 Hz."""
    all_metadata = sorted(Path(out).glob("**/metadata.json"))
    if only_dataset:
        all_metadata = [p for p in all_metadata if p.parent.parent.parent.name == only_dataset]
    if max_clips:
        all_metadata = all_metadata[:max_clips]
    if not all_metadata:
        raise FileNotFoundError(f"No clips under {out}")
    checked = 0
    for path in all_metadata:
        folder = path.parent
        meta = json.loads(path.read_text(encoding="utf-8"))
        fps = int(meta["fps"])
        if fps not in (10, 12, 20):
            raise ValueError(f"{path}: unsupported FPS {fps}")
        if meta.get("frame_count") != N_FRAMES or meta.get("action_count") != N_ACTIONS:
            raise ValueError(f"{path}: incorrect frames/actions")
        a = np.load(folder / "actions_raw_9d.npy", allow_pickle=False)
        p = np.load(action_pose_path(meta, folder), allow_pickle=False)
        ts = np.load(folder / "timestamps_s.npy", allow_pickle=False)
        K_source = np.load(folder / "K_source_per_frame.npy", allow_pickle=False)
        K_video = np.load(folder / "K_video_per_frame.npy", allow_pickle=False)
        distortion = np.load(folder / "distortion_source_per_frame.npy", allow_pickle=False)
        if a.shape != (N_ACTIONS, 9) or p.shape != (N_FRAMES, 4, 4) or ts.shape != (N_FRAMES,):
            raise ValueError(f"{path}: malformed GT tensors")
        if K_source.shape != (N_FRAMES, 3, 3) or K_video.shape != (N_FRAMES, 3, 3):
            raise ValueError(f"{path}: malformed K per frame")
        if distortion.shape[0] != N_FRAMES:
            raise ValueError(f"{path}: malformed distortion array")
        if not np.all(np.diff(ts) > 0):
            raise ValueError(f"{path}: nonmonotonic timestamps")
        target_dt = 1.0 / fps
        tol = {10: 0.07, 12: 0.04, 20: 0.025}[fps]
        max_gap = float(np.max(np.abs(np.diff(ts) - target_dt)))
        if max_gap > tol:
            raise ValueError(f"{path}: dt deviation {max_gap:.5f} > {tol}")
        reconstruction = decode_local(a, "backward_framewise")
        terr = float(np.max(np.linalg.norm(reconstruction[:, :3, 3] - p[:, :3, 3], axis=1)))
        relative_R = np.swapaxes(reconstruction[:, :3, :3], 1, 2) @ p[:, :3, :3]
        c = np.clip((np.trace(relative_R, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
        rerr = float(np.max(np.degrees(np.arccos(c))))
        if terr > 0.002 or rerr > 0.05:
            raise ValueError(f"{path}: action GT roundtrip error {terr:.6f} m / {rerr:.6f} deg")
        if verify_video:
            cap = cv2.VideoCapture(str(folder / "video.mp4"))
            if not cap.isOpened():
                raise ValueError(f"{path}: undecodable mp4")
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            count = 0
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                count += 1
            cap.release()
            if count != N_FRAMES or abs(video_fps - fps) > 0.01:
                raise ValueError(f"{path}: MP4 {count}f@{video_fps}Hz, expected {N_FRAMES}f@{fps}Hz")
            if [width, height] != meta["video_size_wh"]:
                raise ValueError(f"{path}: video size mismatch")
        checked += 1
        print(f"[pass] {meta['clip_id']}: {fps}Hz, dt_err={max_gap:.5f}s, gt_err={terr:.6f}m")
    result = {"verified_clips": checked}
    print("[verified]", result)
    return result


def _iter_sequences(args):
    if args.dataset == "kitti":
        return iter_kitti(Path(args.root), camera=args.camera,
                          sequences=args.sequences,
                          poses_dir=Path(args.poses_dir) if args.poses_dir else None,
                          calib_root=Path(args.calib_root) if args.calib_root else None)
    if args.dataset == "waymo":
        return iter_waymo(Path(args.root), camera=args.camera,
                          max_logs=args.max_logs, compression=args.compression)
    if args.dataset == "nuscenes":
        return iter_nuscenes(Path(args.root), camera=args.camera,
                             max_logs=args.max_logs, strict=args.strict)
    return iter_av2(Path(args.root), camera=args.camera, max_logs=args.max_logs)


def run_prepare(args) -> int:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required")
    if not 0 <= args.crf <= 51 or args.stride <= 0 or args.max_clips < 0:
        raise ValueError("Invalid crf, stride or max-clips")
    if not 0 < args.virtual_hfov_deg < 180:
        raise ValueError("Virtual HFOV must be between 0 and 180 degrees")
    if args.virtual_width <= 0 or args.virtual_height <= 0:
        raise ValueError("Invalid virtual output dimensions")
    # Reject unsupported mode BEFORE scanning potentially large source metadata.
    target_fps(args.dataset, args.variant)
    args.camera = args.camera or DEFAULT_CAMERA[args.dataset]
    out_variant = Path(args.out_root).expanduser().resolve() / args.variant
    if out_variant.exists() and (out_variant / "index.jsonl").exists():
        # Existing clips must match the desired output contract.
        for line in (out_variant / "index.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            meta = json.loads(Path(row["meta"]).read_text(encoding="utf-8"))
            if meta.get("ablation_variant") != args.variant:
                raise ValueError(f"Output root contains another variant: {row['meta']}")
    print(f"[configuration] variant={args.variant} dataset={args.dataset} camera={args.camera} "
          f"output_fps={target_fps(args.dataset, args.variant)}Hz / 61f / 60 actions; "
          f"output_root={out_variant}")
    if target_fps(args.dataset, args.variant) == 20:
        print("[note] 61 frames at 20 Hz cover approximately 3 seconds, not 6 seconds.")
    elif target_fps(args.dataset, args.variant) == 12:
        print("[note] 61 frames at nuScenes native ~12Hz cover approximately 5 seconds; original timestamps are preserved.")
    total = 0
    iterator = _iter_sequences(args)
    for log_number, (sequence, split, frames) in enumerate(iterator):
        if args.dataset == "kitti" and args.max_logs and log_number >= args.max_logs:
            break
        runs = select_runs(frames, args.dataset, args.variant, args.tolerance_ms)
        count = sum(max(0, 1 + (len(run) - N_FRAMES) // args.stride) for run in runs)
        print(f"[sequence] {args.dataset}/{sequence} {split}: source_frames={len(frames)}, "
              f"runs={len(runs)}, candidates={count}")
        if not count:
            continue
        original_order = {id(frame): i for i, frame in enumerate(frames)}
        for run in runs:
            for offset in range(0, len(run) - (N_FRAMES - 1), args.stride):
                selected = run[offset:offset + N_FRAMES]
                if len(selected) != N_FRAMES:
                    continue
                first_index = original_order[id(selected[0])]
                try:
                    result = write_variant_clip(
                        selected, dataset=args.dataset, sequence=sequence,
                        split=split, variant=args.variant, first_index=first_index,
                        out_root=Path(args.out_root).expanduser().resolve(),
                        encoder=args.encoder, crf=args.crf, preset=args.preset,
                        width=args.virtual_width, height=args.virtual_height,
                        hfov=args.virtual_hfov_deg, camera=args.camera,
                        overwrite=args.overwrite)
                    if result is not None:
                        total += 1
                        print("  [created]", result["clip_id"])
                except Exception as exc:
                    print(f"  [error] {sequence}/{first_index}: {exc}")
                    if args.strict:
                        raise
                if args.max_clips and total >= args.max_clips:
                    break
            if args.max_clips and total >= args.max_clips:
                break
        if args.max_clips and total >= args.max_clips:
            break
    indexed = rebuild_index(out_variant)
    print(f"[done] new={total} indexed={indexed} index={out_variant / 'index.jsonl'}")
    return total


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="3-way AV camera intrinsic / FPS ablation")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Prepare KITTI, Waymo, AV2, or nuScenes")
    p.add_argument("--dataset", required=True, choices=list(NOMINAL_SOURCE_FPS))
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--root", required=True)
    p.add_argument("--out-root", required=True, help="One root; each variant gets a separate subdirectory")
    p.add_argument("--camera", default=None)
    p.add_argument("--sequences", nargs="*", default=None)
    p.add_argument("--poses-dir", default=None)
    p.add_argument("--calib-root", default=None)
    p.add_argument("--compression", default="", choices=["", "GZIP"])
    p.add_argument("--max-logs", type=int, default=0)
    p.add_argument("--max-clips", type=int, default=0)
    p.add_argument("--stride", type=int, default=60)
    p.add_argument("--tolerance-ms", type=float, default=35.0,
                   help="10Hz grid nearest-frame tolerance; native mode uses gap checks")
    p.add_argument("--encoder", choices=["auto", "official", "local"], default="auto")
    p.add_argument("--virtual-width", type=int, default=640)
    p.add_argument("--virtual-height", type=int, default=384)
    p.add_argument("--virtual-hfov-deg", type=float, default=45.0)
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--preset", default="veryfast", choices=["ultrafast", "superfast", "veryfast", "fast", "medium"])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--strict", action="store_true")
    p.set_defaults(func=run_prepare)
    v = sub.add_parser("validate", help="Validate preprocessed output at 10, 12, and 20 Hz")
    v.add_argument("--out-root", required=True)
    v.add_argument("--variant", required=True, choices=VARIANTS)
    v.add_argument("--max-clips", type=int, default=10)
    v.add_argument("--only-dataset", choices=list(NOMINAL_SOURCE_FPS), default=None)
    v.add_argument("--no-video-decode", action="store_true")
    v.set_defaults(func=lambda args: validate_clips(Path(args.out_root) / args.variant,
                                                     args.max_clips, not args.no_video_decode,
                                                     args.only_dataset))
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
