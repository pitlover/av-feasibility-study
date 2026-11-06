"""Native Generator AV Inverse Dynamics dataset for NVIDIA Cosmos Framework.

Uses its published ActionTransformPipeline, not an unrelated CNN/GRU proxy.
WARNING: experimentally composed adapter; framework and model weights must be tested on the server.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class AVInverseDataset(Dataset):
    def __init__(self, manifest: str, *, max_dt_error_s: float = 0.07, validate_on_load: bool = True):
        self.items = [json.loads(x) for x in Path(manifest).read_text().splitlines() if x.strip()]
        if not self.items:
            raise ValueError(f'No clips in {manifest}')
        self.max_dt_error_s = float(max_dt_error_s)
        self.validate_on_load = validate_on_load

    def __len__(self):
        return len(self.items)

    def get_shuffle_blocks(self):
        # Compatible with the official ActionIterableShuffleDataset wrapper if desired.
        return [(i, 1) for i in range(len(self.items))]

    def __getitem__(self, idx):
        import cv2
        from cosmos_framework.data.generator.action.utils.domain_utils import get_domain_id
        from .data import preflight

        row = self.items[idx]
        if self.validate_on_load:
            preflight(row, max_dt_error_s=self.max_dt_error_s)
        video_path = row['video']
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise OSError(f'Cannot decode video {video_path}')
        frames = []
        try:
            while len(frames) < 62:
                ok, bgr = cap.read()
                if not ok: break
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        if len(frames) != 61:
            raise ValueError(f'Expected 61 frames, got {len(frames)}: {video_path}')
        rgb = np.stack(frames)                             # T,H,W,C uint8
        video = torch.from_numpy(rgb).permute(3,0,1,2).contiguous() # C,T,H,W uint8
        actions = np.load(row['action_file'], allow_pickle=False)
        if actions.shape != (60,9):
            raise ValueError(f'Expected 60x9 GT action: {row["clip_id"]}')
        return {
            'video': video,
            'action': torch.as_tensor(actions.copy(), dtype=torch.float32),
            'mode': 'inverse_dynamics',
            'domain_id': torch.tensor(get_domain_id('av'), dtype=torch.long),
            'viewpoint': 'ego_view',
            'conditioning_fps': torch.tensor(10, dtype=torch.long),
            'idle_frames': torch.tensor(0, dtype=torch.long),
            'ai_caption': 'You are an autonomous vehicle planning system.',
        }


def get_action_av_sft_dataset(*, manifest: str, tokenizer_config=None, max_action_dim=64,
                              cfg_dropout_rate=0.1, resolution='480', format_prompt_as_json=True,
                              max_dt_error_s=0.07, iterable_shuffle=True, episode_shuffle_seed=42):
    from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline
    from cosmos_framework.data.generator.action.datasets.action_sft_dataset import ActionSFTDataset, ActionIterableShuffleDataset
    raw = AVInverseDataset(manifest, max_dt_error_s=max_dt_error_s)
    transformer = ActionTransformPipeline(
        tokenizer_config=tokenizer_config,
        max_action_dim=max_action_dim,
        cfg_dropout_rate=cfg_dropout_rate,
        append_viewpoint_info=True,
        append_duration_fps_timestamps=True,
        append_resolution_info=True,
        append_idle_frames=False,
        format_prompt_as_json=format_prompt_as_json,
    )
    wrapped = ActionSFTDataset(raw, transformer, resolution)
    return ActionIterableShuffleDataset(wrapped, seed=episode_shuffle_seed) if iterable_shuffle else wrapped
