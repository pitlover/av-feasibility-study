"""Check whether each saved camera pose is consistent with saved extrinsics.

This checks algebraic consistency, not the accuracy of the source calibration
or true physical vehicle-center definition.
"""
import argparse
import numpy as np
from datasets import load_manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--max-sequences', type=int, default=5)
    ap.add_argument('--max-frames', type=int, default=20)
    ap.add_argument('--atol-m', type=float, default=1e-5)
    args = ap.parse_args()

    doc = load_manifest(args.manifest)
    seqs = doc['sequences'][:args.max_sequences]
    if not seqs:
        raise RuntimeError('Manifest has no sequences')

    for s in seqs:
        T_VC = np.asarray(s['T_vehicle_camera'], dtype=np.float64)
        if T_VC.shape != (4,4):
            raise ValueError(f"{s['name']}: bad T_vehicle_camera shape")
        rot_det = np.linalg.det(T_VC[:3,:3])
        if abs(rot_det - 1) > 1e-4:
            raise ValueError(f"{s['name']}: invalid rotation determinant: {rot_det}")
        errors = []
        for frame in s['frames'][:args.max_frames]:
            T_WC = np.asarray(frame['pose'], dtype=np.float64)
            T_WV = np.asarray(frame['vehicle_pose'], dtype=np.float64)
            reconstructed = T_WV @ T_VC
            errors.append(np.max(np.abs(T_WC - reconstructed)))
        e = max(errors) if errors else float('nan')
        print(f"{s['name']} | {s['vehicle_frame']} | "
              f"source={s['extrinsic_source']} | max_SE3_matrix_diff={e:.3g}")
        if not (e < args.atol_m):
            raise AssertionError(f"{s['name']}: Extrinsic transform consistency failed")
    print('OK: camera pose = vehicle pose @ camera-to-reference extrinsic.')


if __name__ == '__main__':
    main()
