"""Shared 10 Hz resampling, video output, per-clip QC and index writer."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .geometry import encode_actions, decode_local, localize, validate_pose_array


@dataclass
class Frame:
    t_s: float
    image: Path | bytes
    world_ego: np.ndarray
    ego_camera: np.ndarray
    K: np.ndarray
    distortion: np.ndarray = field(default_factory=lambda: np.zeros(5))
    source_id: str = ''
    needs_undistort: bool = False


def safe_id(s: str) -> str:
    s = re.sub(r'[^\w\-]+', '_', str(s))
    return s[:140]


def frame_runs_10hz(frames: list[Frame], fps: int = 10, tolerance_s: float = 0.035):
    """Snap source images to a 10Hz time grid; split runs on missing ticks.

    Never duplicate source frames. Never interpolate RGB images. Each run has
    unique, increasing source timestamps. At least 61 frames are needed later.
    """
    frames = sorted(frames, key=lambda f: f.t_s)
    if not frames:
        return []
    ts = np.array([f.t_s for f in frames], np.float64)
    step = 1. / fps
    start = float(ts[0])
    count = int(np.floor((float(ts[-1]) - start) / step + 1e-6)) + 1
    if count > 100_000:
        raise ValueError('Unexpectedly large timeline; split the input into segments first')
    current: list[Frame] = []
    runs: list[list[Frame]] = []
    previous_idx = -1
    for k in range(count):
        target = start + k * step
        i = int(np.searchsorted(ts, target))
        candidates = [j for j in (i - 1, i) if 0 <= j < len(ts) and j > previous_idx]
        j = min(candidates, key=lambda j: abs(ts[j] - target)) if candidates else None
        if j is None or abs(ts[j] - target) > tolerance_s:
            if current:
                runs.append(current)
            current = []
            continue
        current.append(frames[j])
        previous_idx = j
    if current:
        runs.append(current)
    return runs


def frame_runs_kitti_native(frames: list[Frame], min_dt_s: float = 0.04,
                            max_dt_s: float = 0.16):
    """Use KITTI's original frame order for its nominal-10Hz video stream.

    KITTI timestamps need not match a fixed global 0.100-s grid. Requiring
    unique images on such a grid can artificially break every run before
    61 frames, producing zero clips. Keep consecutive source frames without
    duplicating or interpolating RGB, but split genuine time/image gaps.

    Note: exported MP4 is encoded at nominal 10 FPS; timestamp files always
    retain the acquisition times for physical-duration-aware evaluation.
    """
    frames = sorted(frames, key=lambda f: f.t_s)
    if not frames:
        return []
    runs: list[list[Frame]] = []
    run: list[Frame] = [frames[0]]

    def _index(f: Frame):
        try:
            return int(Path(f.source_id).stem)
        except (ValueError, TypeError):
            return None

    for prev, cur in zip(frames[:-1], frames[1:]):
        dt = cur.t_s - prev.t_s
        prev_idx, idx = _index(prev), _index(cur)
        contiguous_index = (prev_idx is None or idx is None or idx == prev_idx + 1)
        if min_dt_s <= dt <= max_dt_s and contiguous_index:
            run.append(cur)
        else:
            runs.append(run)
            run = [cur]
    runs.append(run)
    return runs


def decode_image(frame: Frame):
    if isinstance(frame.image, bytes):
        img = cv2.imdecode(np.frombuffer(frame.image, np.uint8), cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(frame.image), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'Could not decode {frame.source_id}')
    if frame.needs_undistort and np.max(np.abs(frame.distortion)) > 1e-9:
        img = cv2.undistort(img, frame.K, frame.distortion)
    return img


def letterbox(bgr: np.ndarray, width: int, height: int, K: np.ndarray):
    h, w = bgr.shape[:2]
    scale = min(width / w, height / h)
    sw, sh = min(width, round(w * scale)), min(height, round(h * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    im = cv2.resize(bgr, (sw, sh), interpolation=interpolation)
    dx, dy = (width - sw) // 2, (height - sh) // 2
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[dy:dy + sh, dx:dx + sw] = im
    K_out = np.asarray(K, dtype=np.float64).copy()
    K_out[0, 0] *= sw / w
    K_out[1, 1] *= sh / h
    K_out[0, 2] = K[0, 2] * sw / w + dx
    K_out[1, 2] = K[1, 2] * sh / h + dy
    K_out[0, 1] *= sw / w
    return canvas, K_out


def _dump(path: Path, obj: Any):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')



def encode_h264_video(frames: list[Frame], dst: Path, fps: int, width: int,
                      height: int, crf: int = 18, preset: str = 'veryfast') -> np.ndarray:
    """Encode 61 preprocessed frames into a browser-compatible H.264 MP4.

    Pipes BGR24 frames directly to FFmpeg; never creates an intermediary mp4v
    video or loads the full video into memory. Returns output intrinsics.
    """
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        raise RuntimeError('FFmpeg is required for H.264 output. Install ffmpeg '
                           'and ensure `ffmpeg` is visible on PATH.')
    if width % 2 or height % 2:
        raise ValueError('H.264 yuv420p requires even --width and --height')
    cmd = [
        ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s:v', f'{width}x{height}',
        '-r', str(fps), '-i', 'pipe:0', '-an',
        '-c:v', 'libx264', '-preset', preset, '-crf', str(crf),
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(dst),
    ]
    # stderr is placed in a temporary file to prevent pipe-buffer deadlocks.
    with tempfile.TemporaryFile(mode='w+b') as err:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=err)
        K_out = None
        try:
            assert proc.stdin is not None
            for src in frames:
                frame, k = letterbox(decode_image(src), width, height, src.K)
                if K_out is None:
                    K_out = k
                proc.stdin.write(frame.tobytes(order='C'))
            proc.stdin.close()
            ret = proc.wait()
        except Exception as ex:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            err.seek(0)
            details = err.read().decode('utf-8', errors='replace').strip()
            if isinstance(ex, BrokenPipeError):
                raise RuntimeError(f'FFmpeg rejected H.264 input: {details}') from ex
            raise
        err.seek(0)
        details = err.read().decode('utf-8', errors='replace').strip()
        if ret != 0 or not dst.is_file() or dst.stat().st_size == 0:
            raise RuntimeError(f'FFmpeg H.264 encoding failed (exit={ret}): {details}')
    assert K_out is not None
    return K_out


def encode_mp4v_video(frames: list[Frame], dst: Path, fps: int,
                      width: int, height: int) -> np.ndarray:
    """Optional legacy encoder (may not be supported by browser video players)."""
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError('OpenCV MP4V writer unavailable')
    K_out = None
    try:
        for src in frames:
            frame, k = letterbox(decode_image(src), width, height, src.K)
            if K_out is None:
                K_out = k
            writer.write(frame)
    finally:
        writer.release()
    if not dst.is_file() or dst.stat().st_size == 0:
        raise RuntimeError(f'OpenCV failed to write {dst}')
    assert K_out is not None
    return K_out


def write_clip(frames: list[Frame], dataset: str, sequence: str, split: str,
               out_root: Path, first_index: int, args) -> dict | None:
    if len(frames) != 61:
        raise ValueError(f'Expected 61 frames, got {len(frames)}')
    clip_id = f'{safe_id(dataset)}_{safe_id(sequence)}_{first_index:06d}'
    dst = out_root / dataset / split / clip_id
    video_file = dst / 'video.mp4'
    if dst.exists():
        if getattr(args, 'overwrite', False):
            shutil.rmtree(dst)
        elif (dst / 'metadata.json').is_file() and video_file.is_file():
            print(f'  [skip exists] {dst} (use --overwrite to re-encode H.264)')
            return None
        else:
            raise RuntimeError(f'Incomplete existing clip at {dst}; rerun with --overwrite')
    dst.mkdir(parents=True, exist_ok=False)
    try:
        ts = np.array([f.t_s for f in frames], dtype=np.float64)
        world_ego = np.stack([f.world_ego for f in frames]).astype(np.float64)
        ego_cam = np.stack([f.ego_camera for f in frames]).astype(np.float64)
        validate_pose_array(world_ego)
        validate_pose_array(ego_cam)
        ego_local = localize(world_ego)
        world_camera = world_ego @ ego_cam
        camera_local = localize(world_camera)
        actions, encoder_used = encode_actions(ego_local, args.convention, args.encoder)
        if actions.shape != (60, 9):
            raise ValueError(f'Invalid action shape {actions.shape}')
        reconstruction = decode_local(actions, args.convention)
        trans_err = float(np.max(np.linalg.norm(reconstruction[:, :3, 3] - ego_local[:, :3, 3], axis=1)))
        rr = np.swapaxes(reconstruction[:, :3, :3], -1, -2) @ ego_local[:, :3, :3]
        cos = np.clip((np.trace(rr, axis1=1, axis2=2) - 1) / 2, -1, 1)
        rot_err_deg = float(np.max(np.rad2deg(np.arccos(cos))))
        if trans_err > 0.002 or rot_err_deg > 0.05:
            raise ValueError(f'Action roundtrip failed: {trans_err:.6f} m, {rot_err_deg:.6f} deg')
        # Save unnormalized data. Do not assume official model/checkpoint action normalizer.
        np.save(dst / 'timestamps_s.npy', ts)
        np.save(dst / 'poses_world_ego.npy', world_ego)
        np.save(dst / 'poses_local_ego.npy', ego_local)
        np.save(dst / 'poses_local_camera.npy', camera_local)
        np.save(dst / 'T_ego_camera.npy', ego_cam)
        np.save(dst / 'actions_raw_9d.npy', actions)

        video_codec = getattr(args, 'video_codec', 'h264')
        if video_codec == 'h264':
            K_out = encode_h264_video(frames, video_file, args.fps,
                                      args.width, args.height,
                                      crf=getattr(args, 'h264_crf', 18),
                                      preset=getattr(args, 'h264_preset', 'veryfast'))
        elif video_codec == 'mp4v':
            K_out = encode_mp4v_video(frames, video_file, args.fps,
                                      args.width, args.height)
        else:
            raise ValueError(f'Unsupported video codec: {video_codec}')
        meta = dict(
            format='cosmos3_av_preprocess_v1', clip_id=clip_id, dataset=dataset,
            sequence=sequence, split=split, camera=args.camera if hasattr(args, 'camera') else None,
            fps=args.fps, frame_count=61, action_count=60,
            video_codec=('h264' if video_codec == 'h264' else 'mpeg4'),
            video_pixel_format=('yuv420p' if video_codec == 'h264' else 'unknown'),
            video_size_wh=[args.width, args.height],
            image_processing='undistort_if_enabled_then_letterbox',
            pose_frame={'kitti': 'velodyne', 'waymo': 'vehicle', 'av2': 'egovehicle'}[dataset],
            pose_transform_semantics='T_target_source, homogeneous column vector, meters',
            pose_world_file='poses_world_ego.npy', pose_local_file='poses_local_ego.npy',
            pose_camera_local_file='poses_local_camera.npy',
            action_file='actions_raw_9d.npy',
            action_encoding='translation3 + rot6d_column_based',
            action_normalization='none',
            pose_convention=args.convention, action_encoder=encoder_used,
            first_timestamp_s=float(ts[0]), last_timestamp_s=float(ts[-1]),
            actual_capture_duration_s=float(ts[-1] - ts[0]),
            actual_effective_fps=float(60 / (ts[-1] - ts[0])),
            clip_timebase=('kitti_native_nominal_10fps' if dataset == 'kitti' and
                           getattr(args, 'kitti_timing', 'native') == 'native' else '10hz_grid'),
            delta_t_min_s=float(np.diff(ts).min()),
            delta_t_max_s=float(np.diff(ts).max()),
            video_path=str(video_file.resolve()),
            K_source=np.asarray(frames[0].K).tolist(),
            K_video=np.asarray(K_out).tolist(),
            distortion_source=np.asarray(frames[0].distortion).tolist(),
            video_distortion_corrected=bool(frames[0].needs_undistort),
            T_ego_camera_first=np.asarray(ego_cam[0]).tolist(),
            GT_roundtrip_max_translation_m=trans_err,
            GT_roundtrip_max_rotation_deg=rot_err_deg,
            source_frame_ids=[f.source_id for f in frames],
        )
        _dump(dst / 'metadata.json', meta)
        inference = dict(
            action_chunk_size=60, domain_name='av', fps=args.fps,
            image_size=480, model_mode='inverse_dynamics',
            name=clip_id, prompt='You are an autonomous vehicle planning system.',
            seed=0, view_point='ego_view', vision_path=str(video_file.resolve()),
        )
        _dump(dst / 'cosmos_inference.json', inference)
        return dict(clip_id=clip_id, dataset=dataset, sequence=sequence, split=split,
                    video=str(video_file.resolve()), meta=str((dst / 'metadata.json').resolve()),
                    action_file=str((dst / 'actions_raw_9d.npy').resolve()))
    except Exception:
        shutil.rmtree(dst, ignore_errors=True)
        raise


def process_sequence(frames: list[Frame], dataset: str, sequence: str, split: str,
                     args, remaining: int) -> int:
    if not frames:
        print(f'[warn] {dataset}/{sequence}: no usable frames')
        return 0
    ts = np.array([f.t_s for f in sorted(frames, key=lambda x: x.t_s)], dtype=np.float64)
    dt = np.diff(ts)
    if len(dt):
        print(f'[timestamps] median_dt={np.median(dt):.6f}s '
              f'min_dt={dt.min():.6f}s max_dt={dt.max():.6f}s '
              f'effective_fps={1 / np.median(dt):.3f}')
    if dataset == 'kitti' and getattr(args, 'kitti_timing', 'native') == 'native':
        runs = frame_runs_kitti_native(frames)
        time_mode = 'native (nominal 10fps MP4; real timestamps preserved)'
    else:
        runs = frame_runs_10hz(frames, fps=args.fps,
                               tolerance_s=args.tolerance_ms / 1000.)
        time_mode = 'strict 10Hz time grid'
    longest = max((len(run) for run in runs), default=0)
    candidates = sum(max(0, 1 + (len(run) - 61) // args.stride)
                     for run in runs)
    print(f'[time sampling] {time_mode}; runs={len(runs)}, '
          f'longest_run={longest}, clip_candidates={candidates}')
    if not candidates:
        print('[warn] No 61-frame runs: check image-number gaps and timestamp '
              'discontinuities; do not loosen constraints blindly.')
    made = 0
    frame_indices = {id(f): i for i, f in enumerate(frames)}
    for run in runs:
        for offset in range(0, len(run) - 60, args.stride):
            clip = run[offset:offset + 61]
            if len(clip) != 61:
                continue
            idx = frame_indices[id(clip[0])]
            try:
                item = write_clip(clip, dataset, sequence, split, Path(args.out), idx, args)
            except Exception as e:
                print(f'  [clip error] {dataset}/{sequence}/{idx}: {e}')
                if args.strict:
                    raise
                continue
            if item:
                index = Path(args.out) / 'index.jsonl'
                index.parent.mkdir(parents=True, exist_ok=True)
                with index.open('a', encoding='utf8') as f:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
                made += 1
                print(f'  [ok] {item["clip_id"]}')
                if made >= remaining:
                    return made
    return made


def rebuild_index(out_root: Path) -> int:
    """Rebuild index.jsonl from finalized clips; removes stale/duplicate entries.

    Runs once at CLI completion (not per clip). Useful after --overwrite.
    """
    out_root = Path(out_root)
    index = out_root / 'index.jsonl'
    rows = []
    for meta_file in sorted(out_root.glob('*/**/metadata.json')):
        folder = meta_file.parent
        video = folder / 'video.mp4'
        actions = folder / 'actions_raw_9d.npy'
        if not video.is_file() or not actions.is_file():
            continue
        try:
            meta = json.loads(meta_file.read_text(encoding='utf-8'))
            rows.append(dict(
                clip_id=meta['clip_id'], dataset=meta['dataset'],
                sequence=meta['sequence'], split=meta['split'],
                video=str(video.resolve()), meta=str(meta_file.resolve()),
                action_file=str(actions.resolve()),
            ))
        except (KeyError, ValueError) as ex:
            print(f'[index warning] skipping invalid metadata {meta_file}: {ex}')
    out_root.mkdir(parents=True, exist_ok=True)
    temp_path = index.with_suffix('.jsonl.tmp')
    with temp_path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    temp_path.replace(index)
    return len(rows)
