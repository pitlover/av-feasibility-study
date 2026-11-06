"""Cosmos3 qualitative visualization: BEV and camera-projected GT/Predicted paths.

Important:
  * Uses the same .geometry.decode_actions() and metrics() as `score`.
  * Frame i is rendered from the REAL GT camera pose at frame i.
  * All GT/Pred future ego positions live in the SAME clip-local frame.
  * `--ego-height-m` defines the ego origin height above a ground contact point.
    KITTI Velodyne is roof-mounted: auto default is an APPROXIMATION (1.7 m).
  * Projection uses K_video from clip metadata (after letterboxing) and
    T_ego_camera.npy from preprocessing. It is NOT a pixel-ground-truth overlay.
  * No depth/occlusion modeling: projected lines may appear over vehicles.
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

GT_COLOR = (60, 205, 70)      # BGR; GT green
PRED_COLOR = (55, 65, 240)    # BGR; prediction red
WHITE = (255, 255, 255)


def _bev_points(gt_poses, pred_poses, size: int, pad: int = 25):
    """X forward (up), Y left (left); both tracks share one metric scale."""
    def xy(poses):
        xyz = poses[:, :3, 3]
        return np.stack((-xyz[:, 1], -xyz[:, 0]), axis=-1)

    a, b = xy(gt_poses), xy(pred_poses)
    points = np.concatenate((a, b))
    lo, hi = points.min(axis=0), points.max(axis=0)
    center = (lo + hi) * 0.5
    span = np.maximum(hi - lo, 1.0)
    scale = min((size - 2 * pad) / span[0], (size - 2 * pad) / span[1])
    px = np.rint((points - center) * scale + size * 0.5).astype(np.int32)
    return px[:len(a)], px[len(a):]


def _draw_bev(gt_px, pred_px, current, size: int):
    canvas = np.full((size, size, 3), 246, np.uint8)
    cv2.rectangle(canvas, (0, 0), (size - 1, size - 1), (105, 105, 105), 1)
    for q in (0.25, 0.5, 0.75):
        k = int(q * size)
        cv2.line(canvas, (k, 0), (k, size), (223, 223, 223), 1)
        cv2.line(canvas, (0, k), (size, k), (223, 223, 223), 1)
    cv2.putText(canvas, "BEV | ego X forward", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(canvas, "GT green | Pred red", (8, size - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 40, 40), 1, cv2.LINE_AA)

    for px, light, color in (
        (gt_px, (180, 229, 190), GT_COLOR),
        (pred_px, (193, 188, 240), PRED_COLOR),
    ):
        if len(px) > 1:
            cv2.polylines(canvas, [px.reshape(-1, 1, 2)], False, light, 1, cv2.LINE_AA)
        if current > 0:
            cv2.polylines(canvas, [px[:current + 1].reshape(-1, 1, 2)],
                          False, color, 2, cv2.LINE_AA)
        cv2.circle(canvas, tuple(px[0]), 3, (100, 100, 100), -1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(px[current]), 5, color, -1, cv2.LINE_AA)
    return canvas


def load_camera_geometry(meta_path, metadata: dict, image_width: int, image_height: int):
    """Read calibration for the video actually fed to Cosmos (not the raw image)."""
    root = Path(meta_path).parent
    K = np.asarray(metadata.get("K_video"), dtype=np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError(f"{root}: K_video missing/invalid; recheck clip metadata")
    if K[0, 0] <= 0 or K[1, 1] <= 0 or abs(np.linalg.det(K)) < 1e-10:
        raise ValueError(f"{root}: invalid K_video")
    expected_wh = metadata.get("video_size_wh")
    if expected_wh and [image_width, image_height] != list(expected_wh):
        raise ValueError(
            f"Video is {image_width}x{image_height}, but K_video is calibrated "
            f"for {expected_wh}. Do not use unadjusted K on resized video."
        )

    extrinsic_path = root / "T_ego_camera.npy"
    if not extrinsic_path.is_file():
        raise FileNotFoundError(
            f"{extrinsic_path} missing: required for physical image projection"
        )
    T_ec = np.asarray(np.load(extrinsic_path, allow_pickle=False), np.float64)
    if T_ec.shape == (4, 4):  # legacy format: constant extrinsic
        T_ec = np.broadcast_to(T_ec, (61, 4, 4)).copy()
    if T_ec.shape != (61, 4, 4) or not np.isfinite(T_ec).all():
        raise ValueError(f"Bad T_ego_camera.npy shape {T_ec.shape}; expected (61,4,4)")
    if not np.allclose(T_ec[:, 3, :], [0, 0, 0, 1], atol=1e-5):
        raise ValueError("T_ego_camera is not homogeneous")

    distortion = np.asarray(metadata.get("distortion_source", []), dtype=np.float64)
    if (distortion.size and np.max(np.abs(distortion)) > 1e-6
            and not metadata.get("video_distortion_corrected", False)):
        raise ValueError(
            "Video has nonzero lens distortion but was not rectified/undistorted. "
            "Simple pinhole projection is invalid. Regenerate rectified video "
            "or extend renderer with distortion-aware projection."
        )
    return K, T_ec


def _footpoints_world(poses, ego_height_m: float):
    """4x4 pose at each frame maps a foot/contact point from ego into initial world.

    If ego origin is roof-mounted (KITTI Velodyne), use a positive height.
    The footpoint is z = -ego_height_m relative to each ego origin.
    """
    p = np.tile(np.array([0.0, 0.0, -ego_height_m, 1.0]),
                (len(poses), 1))
    return np.einsum("nij,nj->ni", poses, p)[:, :3]


def project_trajectory_to_camera(path_world, gt_pose_current, T_ego_camera_current, K):
    """Project 3D points in the clip-local frame into REAL current camera.

    The same real camera is used for BOTH GT and Pred; do not separately
    transform Pred using a predicted camera pose, which falsely aligns paths.
    """
    T_world_camera = gt_pose_current @ T_ego_camera_current
    T_camera_world = np.linalg.inv(T_world_camera)
    pts = np.asarray(path_world, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"world points must be Nx3, got {pts.shape}")
    if not np.isfinite(pts).all():
        raise ValueError("Non-finite path_world")
    cam = pts @ T_camera_world[:3, :3].T + T_camera_world[:3, 3]
    uvw = cam @ K.T
    uv = np.full((len(cam), 2), np.nan, dtype=np.float64)
    valid = cam[:, 2] > 0.05
    uv[valid] = uvw[valid, :2] / uvw[valid, 2, None]
    return uv, cam[:, 2], cam


def _clip_near_plane(p0, p1, near=0.10):
    """Clip 3D camera-space line segment against Z >= near."""
    z0, z1 = p0[2], p1[2]
    if z0 < near and z1 < near:
        return None
    if z0 < near or z1 < near:
        dz = z1 - z0
        if abs(dz) < 1e-10:
            return None
        alpha = (near - z0) / dz
        clipped = p0 + alpha * (p1 - p0)
        if z0 < near: p0 = clipped
        else: p1 = clipped
    return p0, p1


def _pixel(K, p_cam):
    p = K @ p_cam
    if p[2] <= 0:
        raise ValueError("Point behind camera")
    xy = p[:2] / p[2]
    # Keep cv2.clipLine within safe 32-bit range.
    return tuple(np.rint(np.clip(xy, -1000000, 1000000)).astype(np.int32))


def draw_projected_future(frame, gt, pred, T_ec, K, i: int,
                          ego_height_m: float, horizon_steps: int = 30,
                          line_thickness: int = 3):
    """Overlay reconstructed within-clip motion paths for subsequent frames."""
    H, W = frame.shape[:2]
    overlay = frame.copy()
    camera_pose = gt[i]
    cam_inv = np.linalg.inv(camera_pose @ T_ec[i])
    end = min(60, i + horizon_steps)
    if end <= i:
        return frame, {"gt": 0, "pred": 0}
    def camera_points(poses):
        world = _footpoints_world(poses[i:end+1], ego_height_m)
        return world @ cam_inv[:3, :3].T + cam_inv[:3, 3]
    counts = {}
    for label, poses, color in (
        ("gt", gt, GT_COLOR), ("pred", pred, PRED_COLOR)
    ):
        points = camera_points(poses)
        last_visible = None
        visible_segments = 0
        for p0, p1 in zip(points[:-1], points[1:]):
            clipped = _clip_near_plane(p0.copy(), p1.copy())
            if clipped is None: continue
            try: a, b = (_pixel(K, clipped[0]), _pixel(K, clipped[1]))
            except ValueError: continue
            ok, aa, bb = cv2.clipLine((0, 0, W, H), a, b)
            if ok:
                cv2.line(overlay, aa, bb, color, line_thickness, cv2.LINE_AA)
                last_visible = bb
                visible_segments += 1
        if last_visible is not None:
            cv2.circle(overlay, last_visible, line_thickness + 2,
                       color, -1, cv2.LINE_AA)
        counts[label] = visible_segments
    # Blend colored lines with image (no mask needed: overlay shares base).
    return cv2.addWeighted(overlay, 0.85, frame, 0.15, 0), counts


def _text_block(frame, lines, x=12, y=14, font_scale=.45):
    """Readable semi-transparent overlay, constrained to video width."""
    h, w = frame.shape[:2]
    line_h = 24
    sizes = [cv2.getTextSize(str(line), cv2.FONT_HERSHEY_SIMPLEX,
                             font_scale, 1)[0][0] for line in lines]
    block_w = min(w-x-6, max(sizes)+18)
    block_h = min(h-y-4, 16+len(lines)*line_h)
    if block_w <= 0 or block_h <= 0: return
    overlay = frame.copy()
    cv2.rectangle(overlay, (x,y), (x+block_w,y+block_h),(20,25,30),-1)
    cv2.addWeighted(overlay, .75, frame, .25, 0, frame)
    for k, line in enumerate(lines):
        ty=y+23+k*line_h
        if ty>h-7:break
        cv2.putText(frame, str(line), (x+8,ty),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, WHITE,1,cv2.LINE_AA)


def _encode_h264(frames, destination: Path, fps: float, width: int,
                 height: int, crf: int = 18):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg required for H.264 rendering")
    if width % 2 or height % 2:
        raise ValueError(f"YUV420 H.264 needs even dimensions: {width}x{height}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg","-hide_banner","-loglevel","error","-y",
           "-f","rawvideo","-vcodec","rawvideo","-pix_fmt","bgr24",
           "-s",f"{width}x{height}","-r",str(fps),"-i","-",
           "-an","-c:v","libx264","-preset","veryfast","-crf",str(crf),
           "-pix_fmt","yuv420p","-movflags","+faststart",str(destination)]
    with tempfile.TemporaryFile() as stderr:
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,
                              stdout=subprocess.DEVNULL,stderr=stderr)
        try:
            for f in frames:
                if f.shape != (height,width,3) or f.dtype != np.uint8:
                    raise ValueError(f"Frame shape {f.shape}, expected {height,width,3} uint8")
                proc.stdin.write(f.tobytes(order="C"))
            proc.stdin.close()
            rc=proc.wait()
        except Exception:
            if proc.poll() is None: proc.kill()
            proc.wait()
            raise
        if rc != 0:
            stderr.seek(0)
            raise RuntimeError("FFmpeg failed: "+stderr.read().decode(errors="replace"))
    if not destination.is_file() or destination.stat().st_size==0:
        raise RuntimeError(f"H.264 MP4 was not written: {destination}")


def _side_panel(bev, metrics_row, panel_width, height, label, ego_height, mode):
    panel=np.full((height,panel_width,3), (243,243,243), np.uint8)
    cv2.putText(panel, label[:26], (12,25), cv2.FONT_HERSHEY_SIMPLEX,
                .46,(34,34,34),1,cv2.LINE_AA)
    sz=bev.shape[0]
    left=(panel_width-sz)//2
    top=37
    panel[top:top+sz,left:left+sz] = bev
    y=top+sz+30
    for text in metrics_row:
        if y > height-8:break
        cv2.putText(panel,text,(12,y),cv2.FONT_HERSHEY_SIMPLEX,.45,
                    (45,45,45),1,cv2.LINE_AA)
        y+=23
    if mode=="both" and y+20 < height:
        cv2.putText(panel,f"Footpoint offset: {ego_height:.2f}m",
                    (12,y+4),cv2.FONT_HERSHEY_SIMPLEX,.40,
                    (80,80,80),1,cv2.LINE_AA)
    return panel


def render_clip(row: dict, pred_dir, out_file, *, fps=10.0, crf=18,
                layout="both", ego_height_m=None, horizon_steps=30,
                line_thickness=3, model_name=None):
    if layout not in ("inset","side","projected","both"):
        raise ValueError("layout must be inset, side, projected, or both")
    if horizon_steps<1 or horizon_steps>60:
        raise ValueError("--horizon-steps must be between 1 and 60")
    if ego_height_m is None:
        ego_height_m = 1.7 if row["dataset"]=="kitti" else 0.0
    if not np.isfinite(ego_height_m) or not -5 <= ego_height_m <= 5:
        raise ValueError("--ego-height-m must be finite and in [-5,5]")

    info=verify_row(row)
    gt=localize(info["gt_pose"])
    act, source=load_pred(pred_dir,row["clip_id"])
    pred=decode_actions(act,convention=info["metadata"]["pose_convention"])
    err=metrics(gt,pred)
    cap=cv2.VideoCapture(str(row["video"]))
    if not cap.isOpened():raise RuntimeError(f"Could not open {row['video']}")
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w<320 or h<240:
        cap.release()
        raise ValueError(f"Video too small: {w}x{h}")

    projected=layout in ("projected","both")
    with_bev=layout in ("side","inset","both")
    if projected:
        try: K,T_ec=load_camera_geometry(row["meta"],info["metadata"],w,h)
        except Exception:
            cap.release()
            raise
    else: K,T_ec=None,None

    if layout in ("side","both"):
        panel_width=max(300, int(w*.4))
        if panel_width%2:panel_width+=1
        final_w=w+panel_width
        bev_size=min(panel_width-24,h-180)
        if bev_size<100:
            cap.release()
            raise ValueError("Video height too small for side BEV panel")
    elif layout=="inset":
        panel_width=0
        final_w=w
        bev_size=min(300,int(h*.52),int(w*.38))
    else:
        panel_width=0
        final_w=w
        bev_size=0

    if final_w%2 or h%2:
        cap.release()
        raise ValueError(f"H264 dimensions must be even: {final_w}x{h}")

    if with_bev: gt_px,pr_px=_bev_points(gt,pred,bev_size)

    def frames():
        n=0
        while n<61:
            ok, f=cap.read()
            if not ok:break
            if projected:
                f, visible=draw_projected_future(
                    f,gt,pred,T_ec,K,n,ego_height_m,
                    horizon_steps,line_thickness
                )
            else:visible=None
            rre=(f'{err["rre_deg"][n-1]:.2f} deg' if n else 'N/A')
            rte=(f'{err["rte_m"][n-1]:.3f} m' if n else 'N/A')
            texts=[
                f'{(model_name or row["clip_id"])[:35]} | {n+1:02d}/61',
                f'Step RRE: {rre}',
                f'Step RTE: {rte}',
                f'Position drift: {err["ate_m"][n]:.2f} m',
            ]
            _text_block(f,texts,font_scale=.43 if w<900 else .48)
            if projected:
                cv2.rectangle(f,(8,h-29),(min(410,w-8),h-5),(20,25,30),-1)
                cv2.putText(f,"Projected path: GT green | Pred red",
                            (16,h-12),cv2.FONT_HERSHEY_SIMPLEX,.43,
                            WHITE,1,cv2.LINE_AA)
            if layout=="inset":
                bev=_draw_bev(gt_px,pr_px,n,bev_size)
                f[10:10+bev_size,w-bev_size-10:w-10]=bev
                output=f
            elif layout in ("side","both"):
                bev=_draw_bev(gt_px,pr_px,n,bev_size)
                side=_side_panel(bev,[
                    f"Step RRE: {rre}",
                    f"Step RTE: {rte}",
                    f"Position: {err['ate_m'][n]:.2f} m",
                    f"Rotation: {err['abs_rot_deg'][n]:.2f} deg",
                ],panel_width,h,row["clip_id"],ego_height_m,layout)
                output=np.concatenate([f,side],axis=1)
            else:
                output=f
            yield output
            n+=1
        if n!=61:
            raise ValueError(f"Expected 61 video frames, decoded {n}")

    try:
        _encode_h264(frames(),Path(out_file),fps,final_w,h,crf)
    finally:
        cap.release()
    print(f"[RENDER OK] {row['clip_id']} layout={layout} video={out_file} "
          f"source={source}; height_assumption={ego_height_m:.2f}m")


def render_selection(eval_dir,pred_dir,out_dir,*,clip_id=None,max_clips=0,
                     fps=10.0,crf=18,layout="both",ego_height_m=None,
                     horizon_steps=30,line_thickness=3,model_name=None):
    rows=read_jsonl(Path(eval_dir)/"selection.jsonl")
    if clip_id:
        rows=[r for r in rows if r["clip_id"]==clip_id]
        if not rows:
            raise ValueError(f"Clip {clip_id} is not in eval selection")
    if max_clips>0:rows=rows[:max_clips]
    if not rows:raise ValueError("No evaluation clips selected")
    out=Path(out_dir)
    for row in rows:
        render_clip(row,pred_dir,out/f"{row['clip_id']}_{layout}.mp4",
                    fps=fps,crf=crf,layout=layout,ego_height_m=ego_height_m,
                    horizon_steps=horizon_steps,line_thickness=line_thickness,
                    model_name=model_name)
    print(f"[DONE] {len(rows)} qualitative clip(s) -> {out}")
