"""Reuse existing virtual JPG files from the previous project without reprocessing.

This preserves v2 poses unmodified. It does not repair any v2 calibration issues.
"""
import argparse
import json
from camera import camera_metadata, validate_camera
from build_index import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='Old list-of-sequences manifest')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    with open(args.input, encoding='utf-8') as f:
        old = json.load(f)
    if not isinstance(old, list):
        raise ValueError('Expected old v2 manifest (top-level list)')
    camera = camera_metadata()
    sequences = []
    for s in old:
        if 'K_virtual' not in s:
            raise ValueError(f"{s.get('name')}: no explicit K_virtual; cannot verify camera preprocessing")
        validate_camera({'K_virtual': s['K_virtual'],
                         'virtual_size': s.get('virtual_resolution', camera['virtual_size'])})
        if len(s['images']) != len(s['poses']):
            raise ValueError(f"{s['name']}: length mismatch")
        frames = [dict(prepared_path=str(path), pose=pose,
                       timestamp_s=float(i) / 10.0)
                  for i, (path, pose) in enumerate(zip(s['images'], s['poses']))]
        sequences.append(dict(name=s['name'], split=s['split'], source=s['name'].split('_')[0],
                              frames=frames, K_src=None, distortion=None))
    write_json(args.out, dict(format_version=3, stage='prepared', camera=camera,
                              sequences=sequences))
    print(f'Converted {len(sequences)} legacy sequences -> {args.out}')
    print('NOTE: Legacy timestamps are assumed 10Hz. For research-quality training, rebuild from raw data.')


if __name__ == '__main__':
    main()
