# V2_META_COMPAT_PATCH_20261108_V21
"""CLI: python -m cosmos3prep.cli {kitti,waymo,av2,validate} ..."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
import cv2
import numpy as np

from .datasets import iter_kitti, iter_av2, iter_waymo
from .core import process_sequence, rebuild_index
from .geometry import decode_local, validate_pose_array
from .contracts import action_frame, action_pose_path


def main(argv=None):
    parser = argparse.ArgumentParser(description='Official dataset -> 10Hz/61f Cosmos3 AV ID clips')
    sub = parser.add_subparsers(dest='dataset', required=True)
    for dataset in ('kitti', 'waymo', 'av2'):
        p = sub.add_parser(dataset)
        p.add_argument('--root', required=True, help='Root of the extracted official dataset')
        p.add_argument('--out', required=True, help='Output directory')
        p.add_argument('--fps', type=int, choices=[10], default=10)
        p.add_argument('--width', type=int, default=640)
        p.add_argument('--height', type=int, default=384)
        p.add_argument('--image-mode', choices=['virtual','letterbox'], default='virtual',
                       help='virtual: common pinhole intrinsic without changing extrinsic; letterbox: legacy')
        p.add_argument('--virtual-hfov-deg', type=float, default=45.0)
        p.add_argument('--action-frame', choices=['camera','ego'], default='camera',
                       help='Use camera-relative GT for v4; legacy v3 used ego')
        p.add_argument('--stride', type=int, default=60, help='Sliding stride in 10Hz frames')
        p.add_argument('--tolerance-ms', type=float, default=35., help='Max video timestamp deviation from 10Hz grid')
        p.add_argument('--max-clips', type=int, default=0, help='0 = unlimited; use 1 for smoke test')
        p.add_argument('--max-logs', type=int, default=0, help='0 = all logs/files')
        p.add_argument('--convention', choices=['backward_framewise', 'backward_anchored'],
                       default='backward_framewise')
        p.add_argument('--encoder', choices=['auto', 'official', 'local'], default='auto')
        p.add_argument('--video-codec', choices=['h264', 'mp4v'], default='h264',
                       help='Default H.264 via ffmpeg (browser-compatible); mp4v is legacy')
        p.add_argument('--h264-crf', type=int, default=18,
                       help='H.264 quality (0-51; lower=higher quality, default=18)')
        p.add_argument('--h264-preset', default='veryfast',
                       choices=['ultrafast','superfast','veryfast','faster','fast','medium','slow'],
                       help='FFmpeg libx264 speed/size tradeoff')
        p.add_argument('--overwrite', action='store_true')
        p.add_argument('--strict', action='store_true', help='Raise on first clip error')
        if dataset == 'kitti':
            p.add_argument('--camera', default='image_2', choices=['image_0', 'image_1', 'image_2', 'image_3'])
            p.add_argument('--sequences', nargs='*', help='e.g. 00 01 02')
            p.add_argument('--kitti-timing', choices=['native', 'grid'], default='native',
                           help='native: use consecutive KITTI nominal-10Hz frames, preserving original timestamps. '
                                'grid: require strict 0.1-second sampling (may yield zero clips).')
            p.add_argument('--calib-root', type=Path, default=None,
                           help='Separate KITTI calibration root containing sequences/00/calib.txt '
                                '(or dataset/sequences/00/calib.txt). Default: <root>/sequences.')
            p.add_argument('--poses-dir', type=Path, default=None,
                           help='Separate KITTI GT poses directory containing 00.txt, 01.txt, etc. '
                                '(default: <root>/poses)')
        elif dataset == 'waymo':
            p.add_argument('--camera', default='FRONT')
            p.add_argument('--compression', choices=['', 'GZIP'], default='')
        else:
            p.add_argument('--camera', default='stereo_front_left')

    v = sub.add_parser('validate', help='Validate output clips and action roundtrip')
    v.add_argument('--out', required=True)
    v.add_argument('--max-clips', type=int, default=0)
    args = parser.parse_args(argv)
    if args.dataset == 'validate':
        return validate_output(Path(args.out), args.max_clips)
    if args.stride < 1 or args.width < 2 or args.height < 2:
        parser.error('--stride, --width, --height must be positive')
    if args.image_mode == 'virtual' and not 0 < args.virtual_hfov_deg < 180:
        parser.error('--virtual-hfov-deg must be 0..180')
    if args.video_codec == 'h264' and (args.width % 2 or args.height % 2):
        parser.error('--width and --height must both be even for H.264 yuv420p')
    if args.video_codec == 'h264' and shutil.which('ffmpeg') is None:
        parser.error('H.264 output needs ffmpeg on PATH (install ffmpeg)')
    if not 0 <= args.h264_crf <= 51:
        parser.error('--h264-crf must be in 0..51')
    print(f'[setup] dataset={args.dataset} fps=10 frames=61 convention={args.convention} encoder={args.encoder} video_codec={args.video_codec}')
    if args.dataset == 'kitti':
        iterator = iter_kitti(Path(args.root), args.camera, args.sequences,
                              poses_dir=args.poses_dir, calib_root=args.calib_root)
        if args.max_logs:
            import itertools
            iterator = itertools.islice(iterator, args.max_logs)
    elif args.dataset == 'av2':
        iterator = iter_av2(Path(args.root), args.camera, args.max_logs)
    else:
        iterator = iter_waymo(Path(args.root), args.camera, args.max_logs, args.compression)
    total = 0
    for seq, split, frames in iterator:
        print(f'[sequence] {seq} split={split}, input_frames={len(frames)}')
        remaining = args.max_clips - total if args.max_clips else 10**18
        n = process_sequence(frames, args.dataset, seq, split, args, remaining)
        total += n
        if args.max_clips and total >= args.max_clips:
            break
    indexed = rebuild_index(Path(args.out))
    print(f'[done] {total} new clips; total indexed={indexed}; index={Path(args.out) / "index.jsonl"}')
    if not total:
        print('[hint] 0 NEW clips can also mean outputs already exist. See [skip exists] logs; '
              'use --overwrite to re-encode existing videos with H.264. '
              'Otherwise check clip_candidates/timestamps.')


def validate_output(out: Path, max_clips: int = 0):
    paths = sorted(out.glob('**/metadata.json'))
    if max_clips:
        paths = paths[:max_clips]
    if not paths:
        raise RuntimeError(f'No metadata.json found in {out}')
    passed = 0
    for mp in paths:
        folder = mp.parent
        meta = json.loads(mp.read_text())
        a = np.load(folder / 'actions_raw_9d.npy')
        p = np.load(action_pose_path(meta, folder), allow_pickle=False)
        t = np.load(folder / 'timestamps_s.npy')
        validate_pose_array(p)
        assert a.shape == (60, 9) and p.shape == (61, 4, 4) and t.shape == (61,)
        assert np.isfinite(a).all() and np.isfinite(t).all()
        assert np.all(np.diff(t) > 0)
        assert np.all(np.abs(np.diff(t) - .1) <= 2 * 0.035 + 1e-5)
        pr = decode_local(a, meta['pose_convention'])
        terr = float(np.max(np.linalg.norm(pr[:, :3, 3] - p[:, :3, 3], axis=1)))
        if terr >= 0.002:
            raise ValueError(f'{mp}: {action_frame(meta)} GT roundtrip translation error={terr:.6f} m')
        video = cv2.VideoCapture(str(folder / 'video.mp4'))
        assert video.isOpened(), mp
        fps = video.get(cv2.CAP_PROP_FPS)
        n = 0
        while True:
            ret, _ = video.read()
            if not ret:
                break
            n += 1
        video.release()
        assert n == 61 and abs(fps - 10) < 0.01, (mp, n, fps)
        if meta.get('video_codec') == 'h264' and shutil.which('ffprobe'):
            result = subprocess.run([
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=codec_name,pix_fmt',
                '-of', 'default=noprint_wrappers=1', str(folder / 'video.mp4')
            ], capture_output=True, text=True, check=True)
            assert 'codec_name=h264' in result.stdout, (mp, result.stdout)
            assert 'pix_fmt=yuv420p' in result.stdout, (mp, result.stdout)
        passed += 1
        print(f'[pass] {meta["clip_id"]}, max GT action roundtrip error={terr:.6f} m')
    print(f'[validated] {passed} clips')


if __name__ == '__main__':
    main()
