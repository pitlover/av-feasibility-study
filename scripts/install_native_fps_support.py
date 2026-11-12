#!/usr/bin/env python3
"""Apply strict, idempotent *local-only* 10/20 Hz compatibility edits.

This installer is intended for the exact GitHub branch feat/prepare_nuscenes,
commit 9794a1b02e2bf4d4fee09802038361f9bf06a122. It never uses GitHub APIs,
never runs git, never uploads files, and first verifies every anchor. It creates
backups under .three_variants_backups before changing any source file.

Usage
-----
python scripts/install_native_fps_support.py --repo-root . --check
python scripts/install_native_fps_support.py --repo-root . --apply
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import shutil


# (old, new) text edits, limited to verified files. Fail closed if source differs.
PATCHES: dict[str, list[tuple[str, str]]] = {
    "cosmos3av/data.py": [
        (
            "    if meta.get('fps') != 10 or meta.get('frame_count') != 61 or meta.get('action_count') != 60:\n"
            "        raise ValueError(f'{r[\"clip_id\"]}: expected 10fps/61 video frames/60 actions')",
            "    nominal_fps = int(meta.get('fps', 0))\n"
            "    if nominal_fps not in (10, 20) or meta.get('frame_count') != 61 or meta.get('action_count') != 60:\n"
            "        raise ValueError(f'{r[\"clip_id\"]}: expected 10/20fps, 61 video frames and 60 actions')"
        ),
        (
            "    if float(np.abs(dt - 0.1).max()) > max_dt_error_s:\n"
            "        raise ValueError(f'{r[\"clip_id\"]}: max |dt-0.1|={np.abs(dt - 0.1).max():.4f}s')",
            "    target_dt = 1.0 / nominal_fps\n"
            "    timing_tolerance = min(max_dt_error_s, 0.025) if nominal_fps == 20 else max_dt_error_s\n"
            "    if float(np.abs(dt - target_dt).max()) > timing_tolerance:\n"
            "        raise ValueError(f'{r[\"clip_id\"]}: max |dt-{target_dt:.3f}|={np.abs(dt - target_dt).max():.4f}s')"
        ),
        (
            "        fps = cap.get(cv2.CAP_PROP_FPS)",
            "        video_fps = cap.get(cv2.CAP_PROP_FPS)"
        ),
        (
            "        if cnt != 61 or abs(fps - 10) > .01:\n"
            "            raise ValueError(f'{r[\"clip_id\"]}: MP4 {cnt} frames at {fps}fps')",
            "        if cnt != 61 or abs(video_fps - nominal_fps) > .01:\n"
            "            raise ValueError(f'{r[\"clip_id\"]}: MP4 {cnt} frames at {video_fps}fps (expected {nominal_fps})')"
        ),
        (
            "    return {'max_dt_error_s': float(np.abs(dt - .1).max()), 'mean_dt_s': float(dt.mean())}",
            "    return {'max_dt_error_s': float(np.abs(dt - target_dt).max()), 'mean_dt_s': float(dt.mean())}"
        ),
    ],
    "cosmos3av/framework_dataset.py": [
        (
            "        row = self.items[idx]\n        if self.validate_on_load:",
            "        row = self.items[idx]\n"
            "        meta = json.loads(Path(row['meta']).read_text(encoding='utf8'))\n"
            "        if self.validate_on_load:"
        ),
        (
            "            meta = json.loads(Path(row['meta']).read_text(encoding='utf8'))\n"
            "            if action_frame(meta) != 'camera'",
            "            if action_frame(meta) != 'camera'"
        ),
        (
            "            'conditioning_fps': torch.tensor(10, dtype=torch.long),",
            "            'conditioning_fps': torch.tensor(int(meta['fps']), dtype=torch.long),"
        ),
    ],
    "cosmos3av/experiment.py": [
        (
            "import copy\n",
            "import copy\nimport os\n"
        ),
        (
            "AV_FPS = 10",
            'AV_FPS = int(os.environ.get("COSMOS_AV_FPS", "10"))'
        ),
        (
            '"AV dataset contract is 61 video frames / 60 actions / 10 fps / domain \'av\'; "',
            'f"AV dataset contract is 61 video frames / 60 actions / {AV_FPS} fps / domain \'av\'; "'
        ),
    ],
    "cosmos3av/cli.py": [
        (
            "    env['COSMOS_AV_MANIFEST'] = detail['train_manifest']",
            "    env['COSMOS_AV_MANIFEST'] = detail['train_manifest']\n"
            "    # Set policy/export FPS from REAL clip metadata, never assume 10 Hz.\n"
            "    fps_values = {\n"
            "        int(json.loads(Path(row['meta']).read_text(encoding='utf8'))['fps'])\n"
            "        for line in Path(detail['train_manifest']).read_text(encoding='utf8').splitlines()\n"
            "        if line.strip()\n"
            "        for row in [json.loads(line)]\n"
            "    }\n"
            "    if len(fps_values) != 1 or next(iter(fps_values)) not in (10, 20):\n"
            "        raise ValueError(f'Mixed/unsupported training FPS {fps_values}; create a separate 10Hz or 20Hz run')\n"
            "    env['COSMOS_AV_FPS'] = str(next(iter(fps_values)))\n"
            "    print(f\"[TRAIN FPS] {env['COSMOS_AV_FPS']}Hz\")"
        ),
    ],
    "cosmos3eval/data.py": [
        (
            "    if m.get('fps')!=10 or m.get('frame_count')!=61 or m.get('action_count')!=60:\n"
            "        raise ValueError(f'{cid}: expected 10Hz/61 frames/60 actions')",
            "    if m.get('fps') not in (10,20) or m.get('frame_count')!=61 or m.get('action_count')!=60:\n"
            "        raise ValueError(f'{cid}: expected 10/20Hz, 61 frames and 60 actions')"
        ),
        (
            "            'action_chunk_size':60,'fps':10,'image_size':480,'seed':seed,",
            "            'action_chunk_size':60,'fps':int(json.loads(Path(r['meta']).read_text(encoding='utf8'))['fps']),'image_size':480,'seed':seed,"
        ),
    ],
    "cosmos3prep/cli.py": [
        (
            "        assert np.all(np.abs(np.diff(t) - .1) <= 2 * 0.035 + 1e-5)",
            "        expected_fps = int(meta.get('fps', 10))\n"
            "        assert expected_fps in (10, 20), (mp, expected_fps)\n"
            "        dt_tol = 0.07 if expected_fps == 10 else 0.025\n"
            "        assert np.all(np.abs(np.diff(t) - 1.0 / expected_fps) <= dt_tol + 1e-5)"
        ),
        (
            "        assert n == 61 and abs(fps - 10) < 0.01, (mp, n, fps)",
            "        assert n == 61 and abs(fps - expected_fps) < 0.01, (mp, n, fps)"
        ),
    ],
}


def make_updated(path: str, source: str) -> tuple[str, list[str]]:
    """Return updated source; refuse partially applied or unknown code versions."""
    updates: list[str] = []
    current = source
    for idx, (old, new) in enumerate(PATCHES[path], start=1):
        if new in current:
            updates.append(f"already applied {idx}")
            continue
        found = current.count(old)
        if found != 1:
            raise ValueError(f"{path}: edit #{idx} matches {found} times (expected exactly 1); abort")
        current = current.replace(old, new, 1)
        updates.append(f"apply edit {idx}")
    ast.parse(current, filename=path)
    return current, updates


def process(root: Path, apply: bool) -> None:
    pending: list[tuple[Path, str, str]] = []
    for name in PATCHES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; use feat/prepare_nuscenes checkout")
        raw = path.read_text(encoding="utf-8")
        updated, changes = make_updated(name, raw)
        pending.append((path, raw, updated))
        status = "needs patch" if updated != raw else "already patched"
        print(f"[{status}] {name}: {', '.join(changes)}")
    print(f"[all verified] {len(pending)} source files syntax checked")
    if not apply:
        print("[CHECK ONLY] Re-run with --apply to make local changes (never pushes to GitHub).")
        return
    backup_dir = root / ".three_variants_backups"
    # Verify all edits BEFORE changing anything, then preserve immutable backups.
    for path, raw, updated in pending:
        if updated == raw:
            continue
        backup = backup_dir / path.relative_to(root)
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(updated, encoding="utf-8")
        print("[updated]", path.relative_to(root))
    print("[done] LOCAL changes only; backups:", backup_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch local Cosmos3 AV for 10/20Hz, without git push")
    parser.add_argument("--repo-root", default=".")
    flag = parser.add_mutually_exclusive_group(required=True)
    flag.add_argument("--check", action="store_true")
    flag.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    process(Path(args.repo_root).resolve(), args.apply)


if __name__ == "__main__":
    main()
