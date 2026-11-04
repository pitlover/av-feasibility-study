"""Build light-weight raw-data manifests. No virtual images are written here.

Waymo optional --raw-jpeg-cache extracts ORIGINAL JPEG bytes, not virtualized
images, so the PyTorch environment need not import Waymo proto.
"""
import argparse
import json
import warnings
from pathlib import Path
import numpy as np

from camera import camera_metadata
from geometry import kitti_cam2_from_pose_cam0, waymo_optical_pose, R_WAYMO_CAM_FROM_OPT
from kitti_extrinsics import read_calib, kitti_T_V_C2, camera_to_vehicle_pose
from tfrecord_io import each_record, decode_waymo_frame, get_front


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, separators=(',', ':')), encoding='utf-8')
    tmp.replace(path)


def split_continuous(frames, prefix, split, source, K_src, distortion):
    """Split at missing frames or >150 ms timestamp gaps, never connect discontinuities."""
    groups, chunk = [], []
    for f in frames:
        if chunk and not 0.05 <= (f['timestamp_s'] - chunk[-1]['timestamp_s']) <= 0.15:
            if len(chunk) >= 2:
                groups.append(chunk)
            chunk = []
        chunk.append(f)
    if len(chunk) >= 2:
        groups.append(chunk)
    out = []
    for n, group in enumerate(groups):
        out.append(dict(name=f'{prefix}_{n}', source=source, split=split,
                        K_src=K_src, distortion=distortion, frames=group))
    return out



def resolve_kitti_odometry_calibration(base, seq_id, calib_extra_root=None):
    """Read original P0..P3; load sequence-matched Tr from separate calib, if needed."""
    original_path = base / 'calib.txt'
    calib = read_calib(original_path)
    missing = [p for p in ('P0', 'P1', 'P2', 'P3') if p not in calib]
    if missing:
        raise ValueError(f'{original_path} is missing {missing}')
    if 'Tr' in calib or 'Tr_velo_to_cam' in calib:
        return calib, str(original_path)
    if not calib_extra_root:
        raise ValueError(
            f'{original_path} is missing Tr. Download official data_odometry_calib.zip '
            'and specify --kitti-calib-root pointing to its extracted folder. '
            'P0..P3 support camera-relative GT but not Velodyne reference extrinsics.'
        )
    root = Path(calib_extra_root)
    candidates = [
        root / 'dataset' / 'sequences' / seq_id / 'calib.txt',
        root / 'sequences' / seq_id / 'calib.txt',
        root / seq_id / 'calib.txt',
        root / f'calib_{seq_id}.txt',
    ]
    matches = [p for p in candidates if p.is_file()]
    if not matches:
        raise FileNotFoundError(
            f'Extra KITTI calibration for sequence {seq_id} not found. '
            f'Checked: {[str(p) for p in candidates]}'
        )
    extra_path = matches[0]
    extra = read_calib(extra_path)
    for key in ('P0', 'P1', 'P2', 'P3'):
        if key not in extra or calib[key].shape != extra[key].shape:
            raise ValueError(f'Calibration mismatch: {key} missing or different shape in {extra_path}')
        if not np.allclose(calib[key], extra[key], rtol=1e-6, atol=1e-5):
            raise ValueError(
                f'KITTI {seq_id}: {key} differs between {original_path} and {extra_path}; '
                'refusing to combine potentially mismatched sensor calibrations.'
            )
    tr = extra.get('Tr', extra.get('Tr_velo_to_cam'))
    if tr is None or tr.shape != (12,):
        raise ValueError(f'{extra_path} is also missing a 12-value Tr')
    calib['Tr'] = tr.copy()
    return calib, str(extra_path)


def index_kitti(args):
    root = Path(args.root)
    poses_root = Path(args.poses_root) if args.poses_root else root / 'poses'
    val = {s.strip() for s in args.val_seqs.split(',') if s.strip()}
    seqs = []
    for n in range(11):
        seq_id = f'{n:02d}'
        if args.max_seqs and n >= args.max_seqs:
            break
        base = root / 'sequences' / seq_id
        pose_path = poses_root / (seq_id + '.txt')
        if not base.exists() or not pose_path.exists():
            print(f'SKIP missing KITTI {seq_id} folder or GT pose')
            continue
        calib, extrinsic_calib_path = resolve_kitti_odometry_calibration(
            base, seq_id, getattr(args, 'kitti_calib_root', None)
        )
        P0 = calib['P0'].reshape(3, 4)
        P2 = calib['P2'].reshape(3, 4)
        K = P2[:, :3].tolist()
        T_vehicle_camera, extr_meta = kitti_T_V_C2(
            calib, seq_id,
            vehicle_frame=getattr(args, 'kitti_frame', 'velodyne'),
            imu_calib_root=getattr(args, 'kitti_imu_calib_root', None),
        )
        poses = np.loadtxt(pose_path).reshape(-1, 3, 4)
        paths = sorted((base / 'image_2').glob('*.png'))
        times_file = base / 'times.txt'
        times = np.loadtxt(times_file) if times_file.exists() else np.arange(len(poses)) * .1
        if not times_file.exists():
            warnings.warn(f'KITTI {seq_id} missing times.txt; assuming 10Hz')
        if len(paths) != len(poses) or len(times) != len(poses):
            raise ValueError(f'KITTI {seq_id}: images={len(paths)} poses={len(poses)} timestamps={len(times)}')
        frames = []
        for i, (p, pose, ts) in enumerate(zip(paths, poses, times)):
            if p.stem != f'{i:06d}':
                raise ValueError(f'KITTI {seq_id} frame not contiguous at index {i}: {p}')
            T_WC2 = kitti_cam2_from_pose_cam0(pose, P0, P2)
            T_WV = camera_to_vehicle_pose(T_WC2, T_vehicle_camera)
            frames.append(dict(timestamp_s=float(ts), pose=T_WC2.tolist(),
                               vehicle_pose=T_WV.tolist(), raw_path=str(p.resolve())))
        new_seqs = split_continuous(frames, 'kitti_' + seq_id,
                                    'val' if seq_id in val else 'train',
                                    'kitti', K, [0., 0., 0., 0., 0.])
        for s in new_seqs:
            s['T_vehicle_camera'] = T_vehicle_camera.tolist()
            s.update(extr_meta)
            s['extrinsic_source'] = f'{extrinsic_calib_path}:Tr; P0,P2: {base / "calib.txt"}'
        seqs.extend(new_seqs)
        print(f'KITTI {seq_id}: {len(frames)} frames', flush=True)
    return seqs


def index_waymo(args):
    root = Path(args.root)
    seqs = []
    for folder, split in [('training', 'train'), ('validation', 'val')]:
        files = sorted((root / folder).glob('*.tfrecord'))
        if args.max_files_per_split > 0:
            files = files[:args.max_files_per_split]
        if not files and not args.allow_empty_split:
            print(f'WARNING no Waymo {folder} TFRecords at {root / folder}')
        for path in files:
            frames = []
            K_ref, dist_ref, T_VCam_ref = None, None, None
            skipped = 0
            for offset, raw in each_record(path):
                frame = decode_waymo_frame(raw)
                try:
                    image, cam = get_front(frame)
                    if len(image.pose.transform) != 16 or not image.pose_timestamp:
                        raise ValueError('Camera-capture pose or timestamp missing')
                    if len(cam.extrinsic.transform) != 16 or len(cam.intrinsic) != 9:
                        raise ValueError('Invalid camera calibration')
                    vals = np.asarray(cam.intrinsic, dtype=np.float64)
                    K = np.array([[vals[0], 0, vals[2]], [0, vals[1], vals[3]], [0, 0, 1.]])
                    dist = vals[4:9]
                    if K_ref is None:
                        K_ref, dist_ref = K, dist
                        T_VCam_ref = np.asarray(cam.extrinsic.transform).reshape(4, 4)
                    elif not (np.allclose(K, K_ref, rtol=1e-7, atol=1e-5) and
                              np.allclose(dist, dist_ref, rtol=1e-7, atol=1e-7)):
                        raise ValueError('Camera calibration changed inside a TFRecord')
                    T = waymo_optical_pose(np.asarray(image.pose.transform).reshape(4, 4),
                                           np.asarray(cam.extrinsic.transform).reshape(4, 4))
                    item = dict(timestamp_s=float(image.pose_timestamp), pose=T.tolist(),
                                vehicle_pose=np.asarray(image.pose.transform).reshape(4,4).tolist(),
                                tfrecord=str(path.resolve()), offset=int(offset))
                    if args.raw_jpeg_cache:
                        target = Path(args.raw_jpeg_cache) / folder / path.stem / f'{len(frames):06d}.jpg'
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if not target.exists():
                            target.write_bytes(image.image)  # No decode/remap. Original bytes.
                        item['raw_path'] = str(target.resolve())
                    frames.append(item)
                except ValueError as e:
                    skipped += 1
                    if args.strict:
                        raise ValueError(f'{path}:{offset}: {e}') from e
                    # Guard against connecting frames across omissions by inserting an explicit gap.
                    frames.append(None)
            # Preserve splits even if a frame failed validation.
            run = []
            for entry in frames + [None]:
                if entry is None:
                    if len(run) >= 2:
                        new_seqs = split_continuous(run, 'waymo_' + path.stem + f'_part{len(seqs)}',
                                                    split, 'waymo', K_ref.tolist(), dist_ref.tolist())
                        for s in new_seqs:
                            # 'pose' uses optical frame; extrinsic must use it too.
                            T_cam_from_opt = np.eye(4, dtype=np.float64)
                            T_cam_from_opt[:3, :3] = R_WAYMO_CAM_FROM_OPT
                            s['T_vehicle_camera'] = (T_VCam_ref @ T_cam_from_opt).tolist()
                            s['T_vehicle_camera_raw'] = T_VCam_ref.tolist()
                            s['vehicle_frame'] = 'waymo_vehicle'
                            s['vehicle_frame_note'] = 'Waymo vehicle frame, x forward, y left, z up'
                            s['extrinsic_source'] = 'CameraCalibration.extrinsic + optical-axis conversion'
                        seqs += new_seqs
                    run = []
                else:
                    run.append(entry)
            print(f'Waymo {folder} {path.name}: {sum(f is not None for f in frames)} valid, {skipped} skipped', flush=True)
    return seqs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sp = p.add_subparsers(dest='dataset', required=True)
    k = sp.add_parser('kitti')
    k.add_argument('--root', required=True)
    k.add_argument('--poses-root', default=None)
    k.add_argument('--val-seqs', default='09,10')
    k.add_argument('--kitti-calib-root', default=None,
                   help='Extracted official KITTI data_odometry_calib.zip; use when sequence calib.txt lacks Tr')
    k.add_argument('--max-seqs', type=int, default=0, help='0 means all 00..10')
    k.add_argument('--kitti-frame', choices=['velodyne', 'imu'], default='velodyne',
                   help='velodyne: available from Odometry calib, but NOT vehicle center. '
                        'imu: requires matching KITTI RAW IMU calibration')
    k.add_argument('--kitti-imu-calib-root', default=None,
                   help='Root containing DATE/calib_imu_to_velo.txt or SEQ/calib_imu_to_velo.txt')
    k.add_argument('--out', required=True)
    a = sp.add_parser('av2', help='Argoverse 2 Sensor dataset; 20Hz -> 10Hz camera+pose')
    a.add_argument('--root', required=True, help='Directory containing train/<logid>, val/<logid>')
    a.add_argument('--av2-camera', choices=['stereo_front_left', 'stereo_front_right',
                    'ring_front_center', 'ring_front_left', 'ring_front_right'],
                   default='stereo_front_left',
                   help='Default matches OpenVO; ring_front_center is also supported')
    a.add_argument('--av2-fps', type=int, choices=[10, 20], default=10,
                   help='Use 10 Hz for shared KITTI+Waymo baseline, or 20 Hz ablation')
    a.add_argument('--max-logs-per-split', type=int, default=5, help='0 = all logs')
    a.add_argument('--av2-max-pose-gap-ms', type=float, default=150.0)
    a.add_argument('--strict', action='store_true', help='Fail fast on malformed AV2 log')
    a.add_argument('--out', required=True)
    w = sp.add_parser('waymo')
    w.add_argument('--root', required=True)
    w.add_argument('--max-files-per-split', type=int, default=5, help='0 means all files')
    w.add_argument('--raw-jpeg-cache', default=None,
                   help='Optional: extract original JPEG bytes; online training can then avoid Waymo proto')
    w.add_argument('--allow-empty-split', action='store_true')
    w.add_argument('--strict', action='store_true')
    w.add_argument('--out', required=True)
    args = p.parse_args()
    if args.dataset == 'kitti':
        sequences = index_kitti(args)
    elif args.dataset == 'waymo':
        sequences = index_waymo(args)
    elif args.dataset == 'av2':
        from av2_loader import index_av2
        sequences = index_av2(args)
    else:
        raise ValueError(f'Unsupported dataset: {args.dataset}')
    if not sequences:
        raise RuntimeError('No usable sequences indexed')
    manifest = dict(format_version=3, stage='raw', camera=camera_metadata(), sequences=sequences)
    write_json(args.out, manifest)
    print(f"Wrote {len(sequences)} sequences -> {args.out}")


if __name__ == '__main__':
    main()
