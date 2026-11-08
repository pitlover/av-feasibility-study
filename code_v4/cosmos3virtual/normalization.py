# V2_META_COMPAT_PATCH_20261108_V21
"""Explicit, versioned camera-action normalization for the AV SFT experiments.

Only the *translation* channels are standardized. Rot6D stays unchanged so the
identity rotation remains a valid 6D rotation, and an incompatible normalizer
cannot be silently applied. This is a CUSTOM experiment, not NVIDIA's built-in
action normalization. A checkpoint trained with it needs explicit inversion.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from cosmos3prep.contracts import action_frame

FORMAT = "cosmos3_camera_action_normalizer_v1"


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_normalizer(path, expected_sha256=None):
    path = Path(path).expanduser().resolve()
    if expected_sha256 and file_sha256(path) != expected_sha256:
        raise ValueError(f"Action normalizer hash mismatch: {path}")
    d = json.loads(path.read_text(encoding="utf-8"))
    if d.get("format") != FORMAT or d.get("frame") != "camera" or d.get("method") != "translation3_only":
        raise ValueError(f"Unknown/incompatible action normalizer contract: {path}")
    mean = np.asarray(d.get("mean_9d"), dtype=np.float64)
    scale = np.asarray(d.get("scale_9d"), dtype=np.float64)
    if mean.shape != (9,) or scale.shape != (9,) or not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("Normalizer must have finite 9D mean/scale")
    if (scale <= 0).any() or not np.allclose(mean[3:], 0) or not np.allclose(scale[3:], 1):
        raise ValueError("Rot6D channels must remain unchanged; scales must be positive")
    return d


def transform_actions(actions, normalizer, *, inverse=False):
    a = np.asarray(actions, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != 60 or a.shape[1] not in (9, 64):
        raise ValueError(f"Expected 60x9 or padded 60x64 actions, got {a.shape}")
    if a.shape[1] == 64 and not np.allclose(a[:, 9:], 0, atol=1e-5):
        raise ValueError("Non-zero 64D padding: refuse to normalize/de-normalize")
    mean = np.asarray(normalizer["mean_9d"], np.float64)
    scale = np.asarray(normalizer["scale_9d"], np.float64)
    out = a.copy()
    out[:, :9] = (a[:, :9] * scale + mean) if inverse else ((a[:, :9] - mean) / scale)
    if not np.isfinite(out).all():
        raise ValueError("Non-finite action after normalization")
    return out.astype(np.float32)


def fit_normalizer(processed_root, datasets, out):
    from cosmos3av.data import load_index, preflight
    if not datasets or len(set(datasets)) != len(datasets) or set(datasets) - {"kitti", "waymo"}:
        raise ValueError("Stage 1 normalizer may use KITTI/Waymo train data only")
    selected = [r for r in load_index(Path(processed_root)) if r["split"] == "train" and r["dataset"] in datasets]
    if set(r["dataset"] for r in selected) != set(datasets):
        raise ValueError("Each requested Stage 1 dataset must have train clips")
    s = np.zeros(3); sq = np.zeros(3); count = 0
    for row in selected:
        preflight(row)
        meta = json.loads(Path(row["meta"]).read_text(encoding="utf-8"))
        if action_frame(meta) != "camera":
            raise ValueError(f"Not camera-frame GT: {row['clip_id']}")
        a = np.load(row["action_file"], allow_pickle=False)
        tr = np.asarray(a[:, :3], np.float64)
        s += tr.sum(axis=0)
        sq += (tr * tr).sum(axis=0)
        count += len(tr)
    if not count:
        raise ValueError("Empty Stage 1 action sample")
    mean3 = s / count
    std3 = np.sqrt(np.maximum(sq / count - mean3 * mean3, 0))
    scale3 = np.maximum(std3, .05)  # floor avoids near-zero lateral/vertical amplification
    spec = {
        "format": FORMAT, "frame": "camera", "method": "translation3_only",
        "pose_convention": "backward_framewise", "datasets": sorted(datasets),
        "split": "train", "n_clips": len(selected), "n_actions": count,
        "mean_9d": [*mean3.tolist(), *([0.] * 6)],
        "scale_9d": [*scale3.tolist(), *([1.] * 6)],
        "std_translation3_before_floor": std3.tolist(),
        "warning": "CUSTOM training target transform; only denormalize confirmed normalized model predictions.",
    }
    out = Path(out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(out)
    print(f"[NORMALIZER] {out} sha256={file_sha256(out)}; clips={len(selected)}, actions={count}")
    return spec


def _replace_action(obj, normalizer):
    """Replace one unambiguous action payload, retaining the official JSON wrapper."""
    if isinstance(obj, dict):
        if "action" in obj and obj["action"] is not None:
            obj["action"] = transform_actions(obj["action"], normalizer, inverse=True).tolist()
            return 1
        matches = sum(_replace_action(v, normalizer) for k, v in obj.items() if k in ("content", "outputs", "samples"))
        return matches
    if isinstance(obj, list):
        return sum(_replace_action(v, normalizer) for v in obj)
    return 0


def denormalize_predictions(pred_dir, stage_run, out, *, assume_normalized_output=False):
    """Explicit opt-in only. Never infer model-output scale from action shape."""
    if not assume_normalized_output:
        raise ValueError("Pass --assume-normalized-output only AFTER verifying output normalization")
    run = json.loads((Path(stage_run).expanduser() / "run.json").read_text(encoding="utf-8"))
    path, sha = run.get("action_normalizer"), run.get("action_normalizer_sha256")
    if not path or not sha:
        raise ValueError("This stage run has NO custom normalizer. Predictions are not eligible for inversion.")
    norm = load_normalizer(path, expected_sha256=sha)
    src, dst = Path(pred_dir).expanduser().resolve(), Path(out).expanduser().resolve()
    if src == dst or src in dst.parents or dst in src.parents:
        raise ValueError("Prediction input and output directories must be separate, non-nested locations")
    samples = sorted(src.glob("*/sample_outputs.json"))
    if not samples:
        raise FileNotFoundError(f"No sample_outputs.json under {src}")
    for file in samples:
        payload = json.loads(file.read_text(encoding="utf-8"))
        if _replace_action(payload, norm) != 1:
            raise ValueError(f"Expected precisely one action payload in {file}")
        target = dst / file.parent.name / file.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    (dst / "action_space.json").write_text(json.dumps({
        "action_space": "raw_camera", "source": str(src), "normalizer_sha256": sha,
        "warning": "User asserted model output used the custom normalized action space",
    }, indent=2), encoding="utf-8")
    print(f"[DENORMALIZED] {len(samples)} clips -> {dst}")
    return len(samples)
