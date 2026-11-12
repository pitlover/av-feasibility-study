#!/usr/bin/env python3
"""Merge per-dataset preprocessing indices for one Stage-1 run, without touching input data.

Expected input via ``--spec-json``::
    {"base": "/repo", "processed": "/default/preprocessed", "datasets": ["kitti"],
     "sources": {"kitti": "/override"}, "weights": {"kitti": 1.0}}

For ablation outputs this resolves ROOT/{virtual_native,original_native,original_10hz}/index.jsonl.
It preserves recorded per-clip fps and K; it NEVER resamples data or overwrites clips.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

DATASETS = ("kitti", "waymo", "av2", "nuscenes")
VARIANTS = ("virtual_native", "original_native", "original_10hz")
FPS_VALUES = {10, 12, 20}


def locate_source(source: str, dataset: str, base: str) -> Path:
    path = Path(source).expanduser()
    if not path.is_absolute():
        path = Path(base).expanduser() / path
    path = path.resolve()
    options = [path] + [path / variant for variant in VARIANTS]
    found = []
    for option in options:
        index = option / "index.jsonl"
        if not index.is_file():
            continue
        # Only a source that contains this dataset is eligible. Different variants
        # under the same root must be selected explicitly if more than one exists.
        with index.open(encoding="utf-8") as handle:
            if any(json.loads(line).get("dataset") == dataset for line in handle if line.strip()):
                found.append(option)
    if not found:
        raise FileNotFoundError(
            f"{dataset}: no index.jsonl rows for this dataset under {path}; "
            f"checked direct root and {', '.join(VARIANTS)}. "
            "Check that preprocessing has finished."
        )
    if len(found) != 1:
        raise ValueError(f"{dataset}: ambiguous source {path}; choose an explicit variant folder: {found}")
    return found[0]


def fix_paths(row: dict, root: Path) -> dict:
    row = dict(row)
    clip_folder = root / row["dataset"] / row["split"] / row["clip_id"]
    names = {"video": "video.mp4", "action_file": "actions_raw_9d.npy", "meta": "metadata.json"}
    for key, name in names.items():
        original = Path(row[key]).expanduser()
        relocated = clip_folder / name
        if relocated.is_file():
            # Always prefer the explicitly selected source. An old absolute index
            # path may still be valid but point to a DIFFERENT preprocessing variant.
            row[key] = str(relocated.resolve())
            continue
        if original.is_file() and root == original.resolve().parent.parent.parent.parent:
            row[key] = str(original.resolve())
            continue
        raise FileNotFoundError(
            f"{row['clip_id']}: missing {key} in selected source {relocated}; "
            f"indexed path {original} is not an acceptable fallback"
        )
    return row


def merge(spec: dict, output_dir: Path) -> dict:
    datasets = spec["datasets"]
    if not datasets or len(set(datasets)) != len(datasets) or set(datasets) - set(DATASETS):
        raise ValueError(f"Expected unique nonempty datasets from {DATASETS}: {datasets!r}")
    base = Path(spec["base"]).expanduser().resolve()
    default = spec["processed"]
    sources = spec.get("sources", {})
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "index.jsonl"
    summary = {"datasets": datasets, "resolved_sources": {}, "counts": {}, "fps": {}, "image_modes": {}}
    seen_ids: set[str] = set()
    # Atomic write keeps a partially failed merge from looking valid.
    temp = output_dir / "index.jsonl.incomplete"
    try:
        with temp.open("w", encoding="utf-8") as out:
            for dataset in datasets:
                root = locate_source(sources.get(dataset) or default, dataset, str(base))
                summary["resolved_sources"][dataset] = str(root)
                counts, fps_counts, modes = Counter(), Counter(), Counter()
                sequences = {"train": set(), "val": set()}
                with (root / "index.jsonl").open(encoding="utf-8") as inp:
                    for lineno, line in enumerate(inp, 1):
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if row.get("dataset") != dataset:
                            continue
                        row = fix_paths(row, root)
                        cid = row["clip_id"]
                        if cid in seen_ids:
                            raise ValueError(f"Duplicate clip ID across selected sources: {cid}")
                        seen_ids.add(cid)
                        meta = json.loads(Path(row["meta"]).read_text(encoding="utf-8"))
                        if (meta.get("clip_id"), meta.get("dataset"), meta.get("split")) != (
                            cid, dataset, row["split"]
                        ):
                            raise ValueError(f"{cid}: index disagrees with metadata")
                        fps = meta.get("fps")
                        if fps not in FPS_VALUES or type(fps) is not int:
                            raise ValueError(f"{cid}: unsupported fps={fps}; expected 10, 12, or 20")
                        if dataset == "nuscenes" and fps != 12:
                            raise ValueError(f"{cid}: nuScenes prepared clips must have source-native 12Hz, got {fps}")
                        if (meta.get("frame_count"), meta.get("action_count")) != (61, 60):
                            raise ValueError(f"{cid}: expected 61 RGB frames and 60 actions")
                        if meta.get("action_normalization") != "none":
                            raise ValueError(f"{cid}: source actions must be raw, not normalized")
                        image_mode = meta.get("image_mode", "virtual")
                        counts[row["split"]] += 1
                        if row["split"] in sequences:
                            sequences[row["split"]].add(row["sequence"])
                        fps_counts[fps] += 1
                        modes[image_mode] += 1
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                if not counts.get("train"):
                    raise ValueError(f"{dataset}: no training clips in {root}")
                if sequences["train"] & sequences["val"]:
                    raise ValueError(f"{dataset}: train/val sequence leakage in {root}")
                summary["counts"][dataset] = dict(counts)
                summary["fps"][dataset] = {str(k): v for k,v in sorted(fps_counts.items())}
                summary["image_modes"][dataset] = dict(modes)
                print(f"[source] {dataset}: {root}, split={dict(counts)}, fps={dict(fps_counts)}, image={dict(modes)}")
        temp.replace(output_file)
        (output_dir / "source_spec.json").write_text(json.dumps({**spec, **summary}, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[source-index] wrote {len(seen_ids):,} unique clips to {output_file}")
        return summary
    finally:
        temp.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-json", required=True, help="Resolved JSON from stage1_toml_config.py --format source-json")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args(argv)
    merge(json.loads(args.spec_json), Path(args.outdir).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
