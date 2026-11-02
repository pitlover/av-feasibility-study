"""Same pose labels for materialized and on-the-fly image backends."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from camera import HEIGHT, WIDTH, K_VIRTUAL, validate_camera, virtualize_bgr
from geometry import action_from_transform, relative_poses
from tfrecord_io import read_record, decode_waymo_frame, get_front


def load_manifest(path):
    with open(path, encoding='utf-8') as f:
        obj = json.load(f)
    if not isinstance(obj, dict) or obj.get('format_version') != 3:
        raise ValueError(f'{path}: expected v3 manifest; migrate old v2 with migrate_v2.py')
    validate_camera(obj['camera'])
    return obj


def select_sequences(manifest, split):
    return [seq for seq in manifest['sequences'] if seq['split'] == split]


def _decode_source_image(frame):
    if 'raw_path' in frame:
        image = cv2.imread(frame['raw_path'], cv2.IMREAD_COLOR)
    elif 'tfrecord' in frame:
        record = read_record(frame['tfrecord'], frame['offset'])
        msg = decode_waymo_frame(record)
        im, _ = get_front(msg)
        image = cv2.imdecode(np.frombuffer(im.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    else:
        raise ValueError('Frame has neither raw_path nor TFRecord location')
    if image is None:
        raise IOError(f"Could not decode image from {frame}")
    return image


def load_virtual_bgr(seq, frame, mode, min_valid=.95):
    if mode == 'prepared':
        path = frame.get('prepared_path')
        if not path:
            raise ValueError('This frame has no prepared_path. Run materialize.py first.')
        result = cv2.imread(path, cv2.IMREAD_COLOR)
        if result is None:
            raise IOError(f'Cannot read prepared image {path}')
        if result.shape[:2] != (HEIGHT, WIDTH):
            raise ValueError(f'Wrong image size {result.shape} at {path}')
        return result
    if mode == 'online':
        original = _decode_source_image(frame)
        result, _ = virtualize_bgr(original, seq['K_src'], seq['distortion'], min_valid)
        return result
    raise ValueError(f'Unknown mode {mode}')


def rgb_tensor(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().div_(255.0)


class PairDataset(Dataset):
    """(frame_t, frame_t+1, K*, relative_camera_SE3_label) data samples.

    Returns tensors X[6,H,W], K[3,3], y[6].
    """
    def __init__(self, manifest, split='train', mode='prepared', stride=1):
        if isinstance(manifest, (str, Path)):
            manifest = load_manifest(manifest)
        self.mode = mode
        self.seqs = select_sequences(manifest, split)
        self.samples = []
        self.K = torch.tensor(K_VIRTUAL, dtype=torch.float32)
        for sidx, seq in enumerate(self.seqs):
            frames = seq['frames']
            if len(frames) < 2:
                continue
            P = np.asarray([f['pose'] for f in frames], dtype=np.float64)
            rel = relative_poses(P)
            for i in range(0, len(frames) - 1, stride):
                self.samples.append((sidx, i, action_from_transform(rel[i])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sidx, i, target = self.samples[index]
        seq = self.seqs[sidx]
        a = load_virtual_bgr(seq, seq['frames'][i], self.mode)
        b = load_virtual_bgr(seq, seq['frames'][i + 1], self.mode)
        x = torch.cat((rgb_tensor(a), rgb_tensor(b)), dim=0)
        return x, self.K.clone(), torch.from_numpy(target.copy())
