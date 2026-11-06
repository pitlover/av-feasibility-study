"""Qualitative AV inverse-dynamics rendering.

Draws a *local BEV inset* (not a physical camera-image projection) together
with current GT-vs-predicted trajectory errors. Reuses precisely the same
SE(3) decoder as `cosmos3eval score` to avoid inconsistent conventions.

Input: prepared eval selection, real NVIDIA sample_outputs.json; no GT-as-pred fallback.
Output: H.264/AVC MP4 written by FFmpeg/libx264, playable in common browsers.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .data import read_jsonl, verify_row
from .geometry import decode_actions, metrics, localize
from .predictions import load_pred

GT_COLOR = (65, 210, 85)     # BGR
PRED_COLOR = (75, 80, 240)   # BGR


def _bev_points(gt_poses: np.ndarray, pred_poses: np.ndarray, size: int, pad: int = 25):
    """Use X-forward, Y-left ego axes. Forward is up; left is left on BEV."""
    def xy(poses):
        xyz = poses[:, :3, 3]
        return np.stack((-xyz[:, 1], -xyz[:, 0]), axis=-1)

    gt_xy, pr_xy = xy(gt_poses), xy(pred_poses)
    points = np.concatenate([gt_xy, pr_xy], axis=0)
    lo, hi = points.min(axis=0), points.max(axis=0)
    center = 0.5 * (hi + lo)
    span = np.maximum(hi - lo, 1.0)
    scale = min((size - 2 * pad) / span[0], (size - 2 * pad) / span[1])
    px = np.rint((points - center) * scale + size * .5).astype(np.int32)
    return px[:len(gt_xy)], px[len(gt_xy):]


def _draw_bev(gt_px, pred_px, current, size):
    canvas = np.full((size, size, 3), 247, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (size-1, size-1), (90, 90, 90), 1)
    cv2.putText(canvas, 'BEV (local ego)', (8, 19), cv2.FONT_HERSHEY_SIMPLEX, .48, (30,30,30), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'GT green  Pred red', (8, size-10), cv2.FONT_HERSHEY_SIMPLEX, .41, (45,45,45), 1, cv2.LINE_AA)

    def draw_path(points, upto, base, active):
        if len(points) > 1:
            cv2.polylines(canvas, [points.reshape(-1,1,2)], False, base, 1, cv2.LINE_AA)
        if upto >= 1:
            cv2.polylines(canvas, [points[:upto+1].reshape(-1,1,2)], False, active, 2, cv2.LINE_AA)
        cv2.circle(canvas, tuple(points[0]), 4, (100,100,100), -1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(points[upto]), 5, active, -1, cv2.LINE_AA)

    draw_path(gt_px, current, (185,226,190), GT_COLOR)
    draw_path(pred_px, current, (195,195,241), PRED_COLOR)
    return canvas


def _encode_h264(frames, destination: Path, fps: float, width: int, height: int, crf: int):
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('FFmpeg is required for H.264 rendering (ffmpeg not found on PATH)')
    if width % 2 or height % 2:
        raise ValueError(f'YUV420 H.264 requires even dimensions, got {width}x{height}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    cmd = ['ffmpeg','-hide_banner','-loglevel','error','-y',
        '-f','rawvideo','-vcodec','rawvideo','-pix_fmt','bgr24',
        '-s',f'{width}x{height}','-r',str(fps),'-i','-',
        '-an','-c:v','libx264','-preset','veryfast','-crf',str(crf),
        '-pix_fmt','yuv420p','-movflags','+faststart',str(destination)]
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errors)
        try:
            for frame in frames:
                if frame.shape != (height, width, 3) or frame.dtype != np.uint8:
                    raise ValueError('Invalid frame size or dtype')
                process.stdin.write(frame.tobytes())
            process.stdin.close()
            returncode = process.wait()
        except Exception:
            if process.poll() is None:
                process.kill()
            process.wait()
            raise
        if returncode != 0:
            errors.seek(0)
            raise RuntimeError('FFmpeg render failed: '+errors.read().decode(errors='replace'))
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f'Missing output H.264 MP4: {destination}')


def render_clip(row: dict, pred_dir, out_file, *, fps: float = 10.0, crf: int = 18):
    info = verify_row(row)
    gt = localize(info['gt_pose'])
    actions, source = load_pred(pred_dir, row['clip_id'])
    pred = decode_actions(actions, convention=info['metadata']['pose_convention'])
    m = metrics(gt, pred)
    cap = cv2.VideoCapture(str(row['video']))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open input MP4: {row["video"]}')
    w,h = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    inset_size = min(300, int(h*.55), int(w*.40))
    if inset_size < 120:
        cap.release()
        raise ValueError(f'Input video too small for BEV overlay: {w}x{h}')
    gt_px, pr_px = _bev_points(gt,pred,inset_size)
    x0, y0 = w-inset_size-12, 12

    def frames():
        count=0
        while count < 61:
            ok, frame = cap.read()
            if not ok:
                break
            i=count
            inset=_draw_bev(gt_px,pr_px,i,inset_size)
            frame[y0:y0+inset_size,x0:x0+inset_size]=inset
            step_rre=(f'{m["rre_deg"][i-1]:.3f} deg' if i>0 else 'N/A')
            step_rte=(f'{m["rte_m"][i-1]:.3f} m' if i>0 else 'N/A')
            text=[
                f'{row["clip_id"]} | frame {i+1:02d}/61',
                f'RRE (step): {step_rre}',
                f'RTE (step): {step_rte}',
                f'Position error: {m["ate_m"][i]:.3f} m',
                f'Rotation drift: {m["abs_rot_deg"][i]:.3f} deg',
            ]
            font_scale=.46 if w<900 else .55
            text_h=25
            block_w=min(w-inset_size-36, max(170,int(355*font_scale/.55)))
            cv2.rectangle(frame,(10,10),(block_w,17+text_h*len(text)),(15,20,28),-1)
            for n,line in enumerate(text):
                cv2.putText(frame,line,(17,32+n*text_h),cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,(255,255,255),1,cv2.LINE_AA)
            yield frame
            count+=1
        if count != 61:
            raise ValueError(f'Expected 61 frames from {row["video"]}, got {count}')

    try:
        _encode_h264(frames(),Path(out_file),fps,w,h,crf)
    finally:
        cap.release()
    print(f'[RENDER OK] {row["clip_id"]}: {out_file} | prediction={source}')


def render_selection(eval_dir,pred_dir,out_dir,*,clip_id=None,max_clips=0,fps=10.0,crf=18):
    rows=read_jsonl(Path(eval_dir)/'selection.jsonl')
    if clip_id:
        rows=[r for r in rows if r['clip_id']==clip_id]
        if not rows: raise ValueError(f'Clip {clip_id} is not in evaluation selection')
    if max_clips>0:rows=rows[:max_clips]
    out=Path(out_dir)
    if not rows: raise ValueError('No evaluation clips to render')
    for r in rows:
        render_clip(r,pred_dir,out/f'{r["clip_id"]}_gt_pred.mp4',fps=fps,crf=crf)
    print(f'[DONE] {len(rows)} qualitative clip(s) -> {out}')
