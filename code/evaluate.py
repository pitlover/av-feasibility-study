"""Same 6-second RRE/RTE/ATE evaluation for both data modes.

RRE=per-transition geodesic angle (degrees), RTE=L2 camera translation (m),
ATE=RMSE of 61 camera positions in initial camera frame, no scale alignment.
"""
import argparse

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from camera import K_VIRTUAL, validate_camera
from datasets import load_manifest, select_sequences, load_virtual_bgr, rgb_tensor
from geometry import relative_poses, transform_from_action, accumulate, verify_trajectory
from model import PoseResNet18


def evaluate_sequence(model, seq, mode, clip_frames, device, batch_size=16, max_clips=None):
    frames = seq['frames']
    K = torch.from_numpy(K_VIRTUAL.astype('float32'))
    results = []
    for ci, start in enumerate(range(0, len(frames) - clip_frames + 1, clip_frames - 1)):
        if max_clips is not None and ci >= max_clips:
            break
        chunk = frames[start:start + clip_frames]
        poses = np.asarray([f['pose'] for f in chunk], dtype=np.float64)
        verify_trajectory(poses)
        gt_relative = relative_poses(poses)
        gt_local = np.linalg.inv(poses[0]) @ poses
        images = [rgb_tensor(load_virtual_bgr(seq, f, mode)) for f in chunk]
        pairs = torch.stack([torch.cat([images[i], images[i+1]])
                             for i in range(len(images) - 1)])
        predictions = []
        with torch.inference_mode():
            for ix in range(0, len(pairs), batch_size):
                x = pairs[ix:ix + batch_size].to(device)
                kb = K.to(device).unsqueeze(0).expand(len(x), -1, -1)
                predictions.append(model(x, kb).cpu().numpy())
        actions = np.concatenate(predictions)
        est_relative = np.stack([transform_from_action(a) for a in actions])
        rre = [np.degrees(Rotation.from_matrix(gt_relative[i, :3, :3].T @
                                                est_relative[i, :3, :3]).magnitude())
               for i in range(len(est_relative))]
        rte = np.linalg.norm(gt_relative[:, :3, 3] - est_relative[:, :3, 3], axis=1)
        estimated = accumulate(est_relative)
        ate = np.sqrt(np.mean(np.sum((estimated[:, :3, 3] - gt_local[:, :3, 3]) ** 2, axis=1)))
        results.append([float(np.mean(rre)), float(np.mean(rte)), float(ate)])
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--mode', choices=['prepared', 'online'], required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--split', choices=['train', 'val'], default='val')
    p.add_argument('--clip-frames', type=int, default=61)
    p.add_argument('--max-clips', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=16)
    args = p.parse_args()
    if args.clip_frames < 2 or args.max_clips < 1:
        raise ValueError('Invalid clip settings')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if ckpt.get('architecture') != 'PoseResNet18_v3':
        raise ValueError('This checkpoint was not trained with refactored v3 code')
    validate_camera(ckpt['camera'])
    model = PoseResNet18().to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    manifest = load_manifest(args.manifest)
    results = []
    for seq in select_sequences(manifest, args.split):
        measures = evaluate_sequence(model, seq, args.mode, args.clip_frames, device,
                                     args.batch_size, max_clips=args.max_clips - len(results))
        for m in measures:
            results.append(m)
            print(f'{seq["name"]}: RRE={m[0]:.4f} deg, RTE={m[1]:.4f} m, ATE={m[2]:.4f} m', flush=True)
            if len(results) >= args.max_clips:
                break
        if len(results) >= args.max_clips:
            break
    if not results:
        raise RuntimeError('No sufficiently long validation clips; try --clip-frames 11 for smoke test')
    mean = np.mean(results, axis=0)
    print(f'OVERALL clips={len(results)}, RRE={mean[0]:.5f} deg, '
          f'RTE={mean[1]:.5f} m, ATE={mean[2]:.5f} m', flush=True)


if __name__ == '__main__':
    main()
