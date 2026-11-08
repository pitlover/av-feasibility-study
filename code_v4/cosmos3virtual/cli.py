"""Backward-compatible command-line entrypoint for the successful v2 workflow.

"native" retains source image timestamps. Waymo (nominal 10Hz) and AV2 (20Hz)
are sampled to a 10Hz image grid without RGB interpolation/duplication. KITTI
has a dedicated native-clock path to handle its nominal 10Hz timestamp drift.
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None):
    p = argparse.ArgumentParser(description="V2 compatible virtual-camera AV preprocessing and action tools")
    sub = p.add_subparsers(dest="command", required=True)
    for ds in ("kitti", "waymo", "av2"):
        q = sub.add_parser(ds, help=f"Prepare {ds} virtual camera 61-frame clips")
        q.add_argument("--root", required=True)
        q.add_argument("--out", required=True)
        q.add_argument("--camera", default={"kitti":"image_2", "waymo":"FRONT", "av2":"stereo_front_left"}[ds])
        q.add_argument("--time-mode", choices=["native", "grid"], default="native")
        q.add_argument("--width", type=int, default=640)
        q.add_argument("--height", type=int, default=384)
        q.add_argument("--hfov", type=float, default=45.)
        q.add_argument("--stride", type=int, default=60)
        q.add_argument("--max-logs", type=int, default=0)
        q.add_argument("--max-clips", type=int, default=0)
        q.add_argument("--tolerance-ms", type=float, default=35.)
        q.add_argument("--encoder", choices=["auto", "official", "local"], default="auto")
        q.add_argument("--strict", action="store_true")
        q.add_argument("--overwrite", action="store_true")
        if ds == "kitti":
            q.add_argument("--sequences", nargs="*")
            q.add_argument("--poses-dir")
            q.add_argument("--calib-root")
        if ds == "waymo":
            q.add_argument("--compression", choices=["", "GZIP"], default="")
    q = sub.add_parser("validate")
    q.add_argument("--out", required=True)
    q.add_argument("--max-clips", type=int, default=0)
    q = sub.add_parser("fit-normalizer")
    q.add_argument("--processed-root", required=True)
    q.add_argument("--datasets", nargs="+", required=True, choices=["kitti","waymo"])
    q.add_argument("--out", required=True)
    q = sub.add_parser("denormalize-pred")
    q.add_argument("--pred-dir", required=True)
    q.add_argument("--stage-run", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--assume-normalized-output", action="store_true")
    args = p.parse_args(argv)
    if args.command == "fit-normalizer":
        from .normalization import fit_normalizer
        return fit_normalizer(args.processed_root, args.datasets, args.out)
    if args.command == "denormalize-pred":
        from .normalization import denormalize_predictions
        return denormalize_predictions(args.pred_dir, args.stage_run, args.out,
                                       assume_normalized_output=args.assume_normalized_output)
    from cosmos3prep.cli import main as prepare
    if args.command == "validate":
        a = ["validate", "--out", args.out]
        if args.max_clips:
            a += ["--max-clips", str(args.max_clips)]
        return prepare(a)
    a = [
        args.command, "--root", args.root, "--out", args.out,
        "--camera", args.camera, "--width", str(args.width), "--height", str(args.height),
        "--virtual-hfov-deg", str(args.hfov), "--image-mode", "virtual",
        "--action-frame", "camera", "--convention", "backward_framewise",
        "--stride", str(args.stride), "--tolerance-ms", str(args.tolerance_ms),
        "--max-logs", str(args.max_logs), "--max-clips", str(args.max_clips),
        "--encoder", args.encoder,
    ]
    if args.command == "kitti":
        a += ["--kitti-timing", args.time_mode]
        if args.poses_dir:
            a += ["--poses-dir", args.poses_dir]
        if args.calib_root:
            a += ["--calib-root", args.calib_root]
        if args.sequences:
            a += ["--sequences", *args.sequences]
    elif args.command == "waymo":
        a += ["--compression", args.compression]
    if args.strict:
        a.append("--strict")
    if args.overwrite:
        a.append("--overwrite")
    return prepare(a)


if __name__ == "__main__":
    main()
