#!/usr/bin/env python3
"""Print original vs Cosmos3 480-tier effective calibration for one clip.

Does not claim padding pixels form real camera rays; these transforms apply
only to the *valid original content region* of the video.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

# Matches NVIDIA cosmos-framework's VIDEO_RES_SIZE_INFO['480'] as reviewed.
TIER_480 = {
    '1,1': (640, 640), '4,3': (736, 544), '3,4': (544, 736),
    '16,9': (832, 480), '9,16': (480, 832), '2,3': (512, 768),
}


def effective_K(K, width, height):
    target_w, target_h = min(TIER_480.values(), key=lambda wh: abs(height / width - wh[1] / wh[0]))
    scale = min(target_w / width, target_h / height, 1.0)
    # Source uses int(x + .5) for both resized dimensions.
    resized_w = int(width * scale + .5)
    resized_h = int(height * scale + .5)
    sx, sy = resized_w / width, resized_h / height
    K_model = np.asarray(K, dtype=np.float64).copy()
    K_model[0, :] *= sx
    K_model[1, :] *= sy
    K_model[2, :] = [0, 0, 1]
    return K_model, (target_w, target_h), (resized_w, resized_h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metadata', required=True, help='Path to one output metadata.json')
    args = ap.parse_args()
    m = json.loads(Path(args.metadata).read_text())
    w, h = map(int, m['video_size_wh'])
    K = np.asarray(m['K_video'], dtype=np.float64)
    Km, canvas, resized = effective_K(K, w, h)
    print('dataset/variant:', m['dataset'], m.get('ablation_variant'))
    print('FPS:', m['fps'], 'duration:', m['actual_capture_duration_s'])
    print('video size (W,H):', (w, h))
    print('K_video (valid video content):\n', K)
    print('Cosmos3 480 canvas (W,H):', canvas)
    print('Content after resize (W,H):', resized)
    print('pad right/bottom:', canvas[0]-resized[0], canvas[1]-resized[1])
    print('K_model (ONLY valid content, if K were supplied):\n', Km)
    if m.get('source_pixel_geometry') == 'raw_distorted':
        print('CAUTION: raw lens distortion remains; K by itself is insufficient for full projection.')


if __name__ == '__main__':
    main()
