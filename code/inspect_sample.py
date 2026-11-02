"""Visual sanity check: camera projection and sample metadata (both modes)."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from datasets import load_manifest, select_sequences, load_virtual_bgr
from geometry import relative_poses, verify_trajectory


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--mode', choices=['prepared', 'online'], required=True)
    p.add_argument('--split', choices=['train', 'val'], default='train')
    p.add_argument('--sequence-index', type=int, default=0)
    p.add_argument('--out', default='sanity_virtual.jpg')
    args = p.parse_args()
    manifest = load_manifest(args.manifest)
    seq = select_sequences(manifest, args.split)[args.sequence_index]
    im = load_virtual_bgr(seq, seq['frames'][0], args.mode)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, im)
    P = np.asarray([f['pose'] for f in seq['frames'][:min(61, len(seq['frames']))]])
    trans_error, rot_error = verify_trajectory(P)
    delta = relative_poses(P)
    print(f"Sequence: {seq['name']}, frames={len(seq['frames'])}, stage={manifest['stage']}")
    print(f"Image: {args.out}, size={im.shape[1]}x{im.shape[0]}")
    print('Virtual K:\n', np.asarray(manifest['camera']['K_virtual']))
    print('First 5 camera-relative translations (m):\n', delta[:5, :3, 3])
    print(f'GT relative/integrated consistency: trans={trans_error:.3g} m, rot={rot_error:.3g} rad')


if __name__ == '__main__':
    main()
