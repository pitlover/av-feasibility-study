"""Visualize GT path projected onto real image road + top-down BEV + H264 MP4.

KITTI uses approximate flat road and camera height. Waymo uses vehicle pose and
camera extrinsic saved by build_index.py for ground-reference trajectory.
Both modes share the same virtual RGB images and K* projection geometry.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from camera import K_VIRTUAL
from datasets import load_manifest, load_virtual_bgr, select_sequences


def road_reference_points(seq, frames, dataset, height_m):
    poses = np.asarray([f['pose'] for f in frames], dtype=np.float64)
    if dataset == 'waymo' and 'T_vehicle_camera' in seq and all('vehicle_pose' in f for f in frames):
        TvC = np.asarray(seq['T_vehicle_camera'], dtype=np.float64)
        # Under camera mount (same vehicle XY), vehicle z=0 reference plane.
        veh_point = np.array([TvC[0, 3], TvC[1, 3], 0., 1.])
        return np.asarray([np.asarray(f['vehicle_pose']) @ veh_point for f in frames])[:, :3], 'vehicle-ground-reference'
    up = np.array([0., -1., 0.]) if dataset == 'kitti' else np.array([0., 0., 1.])
    return poses[:, :3, 3] - height_m * up, 'flat-ground-approximation'


def project(points_world, T_WC, K, w, h):
    pc = (np.linalg.inv(T_WC) @ np.concatenate([points_world, np.ones((len(points_world), 1))], axis=1).T).T[:, :3]
    good_depth = pc[:, 2] > .2
    uv = np.full((len(pc), 2), np.nan)
    projected = (K @ pc[good_depth].T).T
    uv[good_depth] = projected[:, :2] / projected[:, 2:3]
    inside = good_depth & (uv[:, 0]>=0) & (uv[:, 0]<w) & (uv[:, 1]>=0) & (uv[:, 1]<h)
    return uv, good_depth, inside


def draw(im, uv, depths, inside, i, fps):
    out = im.copy()
    h, w = out.shape[:2]
    for j in range(i+1, len(uv)-1):
        if not (depths[j] and depths[j+1] and np.isfinite(uv[j]).all() and np.isfinite(uv[j+1]).all()):
            continue
        a = tuple(np.round(uv[j]).astype(int))
        b = tuple(np.round(uv[j+1]).astype(int))
        ok, a, b = cv2.clipLine((0, 0, w, h), a, b)
        if ok:
            cv2.line(out, a, b, (5, 5, 5), 7, cv2.LINE_AA)
            cv2.line(out, a, b, (0, 210, 255), 4, cv2.LINE_AA)
    for j in range(i+1, len(uv)):
        if inside[j]:
            pixel = tuple(np.round(uv[j]).astype(int))
            cv2.circle(out, pixel, 3, (0, 255, 0), -1)
            if j % max(1, int(round(fps))) == 0:
                cv2.putText(out, f'+{(j-i)/fps:.1f}s', (pixel[0]+4, pixel[1]-6),
                            cv2.FONT_HERSHEY_SIMPLEX, .38, (255,255,255), 1)
    cv2.putText(out, f'GT ground trajectory t={i/fps:.1f}s', (10, 23),
                cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 3)
    cv2.putText(out, f'GT ground trajectory t={i/fps:.1f}s', (10, 23),
                cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 1)
    return out


def save_bev(points, first_pose, out):
    T_CW = np.linalg.inv(first_pose)
    camera = (T_CW @ np.c_[points, np.ones(len(points))].T).T[:, :3]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(camera[:, 0], camera[:, 2], '-o', markersize=2.5)
    ax.plot(camera[0,0], camera[0,2], 'r*', markersize=12)
    ax.set(xlabel='Camera right (m)', ylabel='Camera forward (m)', title='GT ground trajectory (BEV)')
    ax.axis('equal')
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def encode_video(seq, frames, points, poses, mode, path, fps, w, h):
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('ffmpeg not found: conda install -c conda-forge ffmpeg')
    cmd = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
           '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s:v', f'{w}x{h}',
           '-r', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
           '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p',
           '-movflags', '+faststart', str(path)]
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=err)
        try:
            for i, frame in enumerate(frames):
                img = load_virtual_bgr(seq, frame, mode)
                uv, depths, inside = project(points, poses[i], K_VIRTUAL, w, h)
                overlay = draw(img, uv, depths, inside, i, fps)
                proc.stdin.write(np.ascontiguousarray(overlay).tobytes())
            proc.stdin.close()
            code = proc.wait()
        except Exception:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
            proc.kill()
            proc.wait()
            path.unlink(missing_ok=True)
            raise
        if code != 0:
            err.seek(0)
            raise RuntimeError(err.read().decode('utf-8', errors='replace')[-2000:])
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError('Video output empty')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--mode', choices=['prepared','online'], required=True)
    p.add_argument('--split', choices=['train','val'], default='val')
    p.add_argument('--sequence-index', type=int, default=0)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--frames', type=int, default=61)
    p.add_argument('--camera-height', type=float, default=None)
    p.add_argument('--fps', type=float, default=10.)
    p.add_argument('--outdir', default='viz')
    p.add_argument('--video', action='store_true')
    args = p.parse_args()
    manifest = load_manifest(args.manifest)
    seq = select_sequences(manifest, args.split)[args.sequence_index]
    frames = seq['frames'][args.start: args.start+args.frames]
    if len(frames) < 2:
        raise ValueError('Need at least 2 contiguous frames')
    poses = np.asarray([f['pose'] for f in frames], dtype=np.float64)
    dataset = seq['source']
    height = args.camera_height if args.camera_height is not None else (1.65 if dataset=='kitti' else 2.115)
    if height <= 0:
        raise ValueError('camera height must be positive')
    points, method = road_reference_points(seq, frames, dataset, height)
    first = load_virtual_bgr(seq, frames[0], args.mode)
    h,w=first.shape[:2]
    uv, depths, inside = project(points, poses[0], K_VIRTUAL, w, h)
    output=Path(args.outdir)
    output.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(output/'projection_on_road.png'), draw(first,uv,depths,inside,0,args.fps))
    save_bev(points,poses[0],output/'bev.png')
    print(f'Mode={args.mode}, method={method}, visible points={inside.sum()}/{len(frames)}')
    if args.video:
        encode_video(seq,frames,points,poses,args.mode,output/'projection_video.mp4',args.fps,w,h)
    print(f'Saved to {output}')


if __name__ == '__main__':
    main()
