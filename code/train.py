"""Train balanced KITTI+Waymo 2-frame motion model with --mode prepared|online."""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler

from camera import camera_metadata
from datasets import PairDataset
from model import PoseResNet18


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['prepared', 'online'], required=True)
    p.add_argument('--kitti', required=True, help='v3 manifest (prepared or raw)')
    p.add_argument('--waymo', required=True, help='v3 manifest (prepared or raw)')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--steps-per-epoch', type=int, default=300)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default='checkpoints/resnet18_pose.pt')
    p.add_argument('--resume', default=None)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    kitti = PairDataset(args.kitti, split='train', mode=args.mode)
    waymo = PairDataset(args.waymo, split='train', mode=args.mode)
    if not kitti or not waymo:
        raise RuntimeError(f'Empty train samples: KITTI={len(kitti)}, Waymo={len(waymo)}')
    concat = ConcatDataset([kitti, waymo])
    weights = [0.5 / len(kitti)] * len(kitti) + [0.5 / len(waymo)] * len(waymo)
    sampler = WeightedRandomSampler(weights, args.steps_per_epoch * args.batch_size,
                                    replacement=True)
    loader = DataLoader(concat, batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PoseResNet18().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=True)
        model.load_state_dict(ckpt['model'])
        optim.load_state_dict(ckpt['optimizer'])
        start_epoch = int(ckpt['epoch']) + 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f'mode={args.mode}, KITTI pairs={len(kitti)}, Waymo pairs={len(waymo)}, device={device}', flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        losses = []
        for step, (x, K, target) in enumerate(loader, start=1):
            pred = model(x.to(device), K.to(device))
            target = target.to(device)
            loss_t = torch.nn.functional.smooth_l1_loss(pred[:, :3], target[:, :3])
            loss_r = torch.nn.functional.smooth_l1_loss(pred[:, 3:], target[:, 3:])
            loss = loss_t + 2.0 * loss_r
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            losses.append(float(loss.detach().cpu()))
            if step % 50 == 0:
                print(f'epoch {epoch} step {step}/{len(loader)} loss={np.mean(losses[-50:]):.5f}', flush=True)
        payload = dict(epoch=epoch, model=model.state_dict(), optimizer=optim.state_dict(),
                       architecture='PoseResNet18_v3', camera=camera_metadata(),
                       input_type='two_frame_rgb_K', output_type='camera_relative_se3_axis_angle',
                       train_mode=args.mode)
        torch.save(payload, out)
        print(f'epoch {epoch}: avg loss {np.mean(losses):.6f}; saved {out}', flush=True)


if __name__ == '__main__':
    main()
