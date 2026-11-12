#!/usr/bin/env python3
"""Summarize ablation outputs without needing Waymo/AV2 SDKs."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

VARIANTS = ("original_10hz", "original_native", "virtual_native")


def summarize(root: Path) -> None:
    for variant in VARIANTS:
        folder = root / variant
        idx = folder / "index.jsonl"
        if not idx.is_file():
            print(f"\n{variant}: no index at {idx}")
            continue
        rows = [json.loads(x) for x in idx.read_text(encoding="utf-8").splitlines() if x.strip()]
        c = Counter()
        fov = defaultdict(list)
        fps_eff = defaultdict(list)
        duration = defaultdict(list)
        for row in rows:
            meta = json.loads(Path(row["meta"]).read_text(encoding="utf-8"))
            key = (meta["dataset"], meta["split"], int(meta["fps"]))
            if meta.get("ablation_variant") != variant:
                raise ValueError(f"Mixed variants: {meta['clip_id']}")
            c[key] += 1
            k = np.asarray(meta["K_video"], dtype=np.float64)
            w = float(meta["video_size_wh"][0])
            hfov = float(np.degrees(2 * np.arctan(w / (2 * k[0, 0]))))
            fov[key].append(hfov)
            fps_eff[key].append(meta["actual_effective_fps"])
            duration[key].append(meta["actual_capture_duration_s"])
        print(f"\n[{variant}] total_clips={len(rows)}")
        print("dataset/split    encoded Hz    clips    seconds(median)  actualHz(median)  HFOV(deg, median)")
        for key in sorted(c):
            ds, split, nominal = key
            med_d = np.median(duration[key]); med_fps = np.median(fps_eff[key])
            med_hfov = np.median(fov[key])
            print(f"{ds}/{split:<12} {nominal:<12} {c[key]:<8} {med_d:<16.3f} {med_fps:<17.3f} {med_hfov:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()
    summarize(Path(args.out_root).expanduser().resolve())


if __name__ == "__main__":
    main()
