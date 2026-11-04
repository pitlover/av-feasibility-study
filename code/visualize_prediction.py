"""Visualize ResNet18 camera-motion predictions against GT on KITTI/Waymo/AV2.

Outputs: bev_comparison.png (camera centers), first_frame_comparison.png
(ground-reference projections), trajectory_comparison.mp4 (optional H264),
trajectory.csv, metrics.json, poses_gt.npy, poses_pred.npy.

Conventions: camera pose T_WC maps optical camera -> world. Predictions and GT
are integrated in the first camera frame. 3D points projected on later actual
frames are transformed by inverse(GT current camera pose), NOT inverse(pred).
Road overlays are approximations, not lidar/road-mesh ground truth.
"""
import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from camera import K_VIRTUAL, WIDTH, HEIGHT, validate_camera
from datasets import load_manifest, load_virtual_bgr, rgb_tensor, select_sequences
from geometry import accumulate, relative_poses, transform_from_action, verify_trajectory
from model import PoseResNet18

GREEN = (50, 190, 55)  # BGR: GT
ORANGE = (0, 140, 255)  # BGR: Pred


def infer(model, seq, frames, mode, device, batch_size):
    images = [rgb_tensor(load_virtual_bgr(seq, f, mode)) for f in frames]
    k = torch.from_numpy(K_VIRTUAL.astype(np.float32)).to(device)
    out = []
    with torch.inference_mode():
        for start in range(0, len(images) - 1, batch_size):
            stop = min(start + batch_size, len(images) - 1)
            pairs = torch.stack([
                torch.cat([images[i], images[i + 1]], dim=0)
                for i in range(start, stop)
            ]).to(device)
            pred = model(pairs, k.unsqueeze(0).expand(len(pairs), -1, -1))
            out.append(pred.cpu().numpy())
    return np.concatenate(out)


def ground_point_in_camera(seq, height):
    """Point below the mounting location in local optical-camera coords."""
    if seq.get('source') in ('waymo', 'av2') and 'T_vehicle_camera' in seq:
        t_vc = np.asarray(seq['T_vehicle_camera'], dtype=np.float64)
        if t_vc.shape != (4, 4):
            raise ValueError('Invalid T_vehicle_camera')
        # At vehicle z=0, directly below camera mount: XY coordinates of camera.
        under_mount_vehicle = np.array([t_vc[0, 3], t_vc[1, 3], 0.0, 1.0])
        return (np.linalg.inv(t_vc) @ under_mount_vehicle)[:3], 'vehicle-z0-under-camera'
    # Optical camera axes: x right, y down, z forward.
    return np.array([0.0, height, 0.0]), 'flat-road-camera-height-approximation'


def ground_positions(local_poses, camera_point):
    q = np.r_[camera_point, 1.]
    return (local_poses @ q)[:, :3]


def project_on_frame(points_local, gt_camera_local, K, w, h):
    """Project points in first camera frame onto actual observed camera at t."""
    homo = np.c_[points_local, np.ones(len(points_local))]
    pts = (np.linalg.inv(gt_camera_local) @ homo.T).T[:, :3]
    valid_depth = pts[:, 2] > 0.25
    uv = np.full((len(pts), 2), np.nan, dtype=np.float64)
    proj = (K @ pts[valid_depth].T).T
    uv[valid_depth] = proj[:, :2] / proj[:, 2:3]
    inside = valid_depth & np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return uv, valid_depth, inside


def draw_path(frame, points_local, gt_camera_local, K, start, color, label):
    h, w = frame.shape[:2]
    uv, depth, inside = project_on_frame(points_local, gt_camera_local, K, w, h)
    for j in range(start + 1, len(uv) - 1):
        if not (depth[j] and depth[j + 1]):
            continue
        if not (np.isfinite(uv[j]).all() and np.isfinite(uv[j + 1]).all()):
            continue
        a = tuple(np.rint(uv[j]).astype(int))
        b = tuple(np.rint(uv[j+1]).astype(int))
        ok, a, b = cv2.clipLine((0, 0, w, h), a, b)
        if ok:
            cv2.line(frame, a, b, (20, 20, 20), 5, cv2.LINE_AA)
            cv2.line(frame, a, b, color, 3, cv2.LINE_AA)
    for j in range(start+1, len(uv)):
        if inside[j] and (j-start) % 10 == 0:
            xy = tuple(np.rint(uv[j]).astype(int))
            cv2.circle(frame, xy, 3, color, -1, cv2.LINE_AA)
    return frame


def overlay(image, gt_points, pred_points, gt_local_at_t, frame_idx, metric):
    image = image.copy()
    draw_path(image, gt_points, gt_local_at_t, K_VIRTUAL, frame_idx, GREEN, 'GT')
    draw_path(image, pred_points, gt_local_at_t, K_VIRTUAL, frame_idx, ORANGE, 'Pred')
    cv2.rectangle(image, (0, 0), (image.shape[1], 49), (22, 25, 28), -1)
    cv2.line(image, (12, 16), (38, 16), GREEN, 4)
    cv2.putText(image, 'GT', (44, 20), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(image, (108, 16), (136, 16), ORANGE, 4)
    cv2.putText(image, 'Prediction', (143, 20), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, f'frame={frame_idx:03d} | position error={metric:.3f}m',
                (12, 42), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def save_bev(gt_local, pred_local, output):
    gt = gt_local[:, :3, 3]
    pred = pred_local[:, :3, 3]
    fig, ax = plt.subplots(figsize=(7, 7))
    # Optical X right, optical Z forward; this is camera-center BEV, not road-footprint BEV.
    ax.plot(gt[:, 0], gt[:, 2], '-', color='#239858', lw=2.5, label='GT camera trajectory')
    ax.plot(pred[:, 0], pred[:, 2], '--', color='#e38b2a', lw=2.5, label='Pred camera trajectory')
    ax.scatter([0.0], [0.0], marker='*', s=140, color='black', zorder=3)
    ax.scatter(gt[-1, 0], gt[-1, 2], color='#239858', s=60)
    ax.scatter(pred[-1, 0], pred[-1, 2], color='#e38b2a', s=60)
    ax.set_xlabel('Initial camera right (m)')
    ax.set_ylabel('Initial camera forward (m)')
    ax.set_title('GT vs Prediction — integrated camera trajectory')
    ax.axis('equal')
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def metric_values(gt_rel, pred_rel, gt_local, pred_local):
    errs_r = np.degrees(Rotation.from_matrix(
        np.transpose(gt_rel[:, :3, :3], (0, 2, 1)) @ pred_rel[:, :3, :3]
    ).magnitude())
    errs_t = np.linalg.norm(gt_rel[:, :3, 3] - pred_rel[:, :3, 3], axis=-1)
    pos_errors = np.linalg.norm(gt_local[:, :3, 3] - pred_local[:, :3, 3], axis=1)
    return {'RRE_deg': float(errs_r.mean()),
            'RTE_m': float(errs_t.mean()),
            'ATE_m': float(np.sqrt(np.mean(pos_errors ** 2))),
            'final_position_error_m': float(pos_errors[-1]),
            'per_frame_position_error_m': pos_errors.tolist(),
            'per_transition_RRE_deg': errs_r.tolist(),
            'per_transition_RTE_m': errs_t.tolist()}


def video_encode(out_path, frames, seq, mode, gt_ground, pred_ground, gt_local, position_errors, fps):
    exe = shutil.which('ffmpeg')
    if exe is None:
        raise RuntimeError('ffmpeg not installed. conda install -c conda-forge ffmpeg')
    cmd = [exe, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
           '-pix_fmt', 'bgr24', '-s:v', f'{WIDTH}x{HEIGHT}', '-r', str(fps),
           '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'veryfast',
           '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for i, f in enumerate(frames):
            image = load_virtual_bgr(seq, f, mode)
            vis = overlay(image, gt_ground, pred_ground, gt_local[i], i, position_errors[i])
            proc.stdin.write(np.ascontiguousarray(vis).tobytes())
        proc.stdin.close()
        err = proc.stderr.read()
        code = proc.wait()
        if code != 0:
            raise RuntimeError(f'FFmpeg exited {code}: {err.decode(errors="replace")[-1600:]}')
    except BaseException:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        out_path.unlink(missing_ok=True)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, help='v3 Prepared Manifest or Raw Index')
    p.add_argument('--checkpoint', required=True, help='v3 PoseResNet18 checkpoint')
    p.add_argument('--mode', choices=['prepared', 'online'], required=True)
    p.add_argument('--split', default='val', choices=['train', 'val'])
    p.add_argument('--sequence-index', type=int, default=0)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--frames', type=int, default=61)
    p.add_argument('--batch-size', type=int, default=12)
    p.add_argument('--camera-height', type=float, default=1.65,
                   help='KITTI approximate camera height in meters for road overlay')
    p.add_argument('--fps', type=float, default=10.)
    p.add_argument('--outdir', default='viz_gt_pred')
    p.add_argument('--video', action='store_true')
    args = p.parse_args()
    if args.frames < 2 or args.start < 0 or args.batch_size < 1:
        p.error('frames must >=2; start must >=0; batch-size must >=1')
    if args.camera_height <= 0:
        p.error('camera-height must be positive')
    manifest = load_manifest(args.manifest)
    seqs = select_sequences(manifest, args.split)
    if not seqs:
        raise ValueError(f'No sequences available for split={args.split}')
    if not (0 <= args.sequence_index < len(seqs)):
        raise IndexError(f'sequence-index out of range [0,{len(seqs)-1}]')
    seq = seqs[args.sequence_index]
    frames = seq['frames'][args.start:args.start+args.frames]
    if len(frames) != args.frames:
        raise ValueError(f'Not enough frames in {seq["name"]}: expected {args.frames}, got {len(frames)}')
    timestamps = np.array([f.get('timestamp_s', np.nan) for f in frames], dtype=float)
    if np.isfinite(timestamps).all():
        dts = np.diff(timestamps)
        if not np.all(dts > 0):
            raise ValueError('Non-monotonic timestamp in clip')
        duration = timestamps[-1] - timestamps[0]
        print(f'clip duration={duration:.3f}s; median dt={np.median(dts):.4f}s')
        if abs(duration - (len(frames)-1)/args.fps) > 0.15:
            print('WARNING: clip duration differs from nominal fps; check 10Hz sampling.')
    poses = np.asarray([f['pose'] for f in frames], dtype=np.float64)
    verify_trajectory(poses)
    gt_local = np.linalg.inv(poses[0]) @ poses
    gt_rel = relative_poses(poses)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if ckpt.get('architecture') != 'PoseResNet18_v3':
        raise ValueError('Expected v3 checkpoint (PoseResNet18_v3)')
    validate_camera(ckpt['camera'])
    model = PoseResNet18().to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    actions = infer(model, seq, frames, args.mode, device, args.batch_size)
    pred_rel = np.stack([transform_from_action(a) for a in actions])
    pred_local = accumulate(pred_rel)
    metrics = metric_values(gt_rel, pred_rel, gt_local, pred_local)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out/'poses_gt.npy', gt_local)
    np.save(out/'poses_pred.npy', pred_local)
    np.save(out/'pred_actions_6d.npy', actions)
    (out/'metrics.json').write_text(json.dumps({k:v for k,v in metrics.items() if not k.startswith('per_')}, indent=2))
    with (out/'trajectory.csv').open('w', newline='') as fh:
        fieldnames = ['frame','timestamp_s','gt_x','gt_y','gt_z','pred_x','pred_y','pred_z','position_error_m']
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(frames)):
            g, e = gt_local[i, :3, 3], pred_local[i, :3, 3]
            writer.writerow(dict(frame=i, timestamp_s=timestamps[i],
                                 gt_x=g[0],gt_y=g[1],gt_z=g[2],
                                 pred_x=e[0],pred_y=e[1],pred_z=e[2],
                                 position_error_m=metrics['per_frame_position_error_m'][i]))
    save_bev(gt_local, pred_local, out/'bev_comparison.png')
    q, ground_ref = ground_point_in_camera(seq, args.camera_height)
    gt_ground = ground_positions(gt_local, q)
    pred_ground = ground_positions(pred_local, q)
    first = load_virtual_bgr(seq, frames[0], args.mode)
    cv2.imwrite(str(out/'first_frame_comparison.png'), overlay(
        first, gt_ground, pred_ground, gt_local[0], 0, metrics['per_frame_position_error_m'][0]))
    if args.video:
        video_encode(out/'trajectory_comparison.mp4', frames, seq, args.mode,
                     gt_ground, pred_ground, gt_local,
                     metrics['per_frame_position_error_m'], args.fps)
    print(f'sequence={seq["name"]}, {len(frames)} frames, mode={args.mode}, ground={ground_ref}')
    print(f'RRE={metrics["RRE_deg"]:.4f}deg, RTE={metrics["RTE_m"]:.4f}m, '
          f'ATE={metrics["ATE_m"]:.4f}m, final_position_error={metrics["final_position_error_m"]:.4f}m')
    print(f'Saved to {out.resolve()}')


if __name__ == '__main__':
    main()
