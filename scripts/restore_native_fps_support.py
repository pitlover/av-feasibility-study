#!/usr/bin/env python3
"""Restore ONLY local backups saved by install_native_fps_support.py.

Default --check lists files that would be restored. --restore explicitly
replaces CURRENT local sources with saved pre-patch files.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from install_native_fps_support import PATCHES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-root', default='.')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--restore', action='store_true')
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    backup_dir = root / '.three_variants_backups'
    found = []
    for name in PATCHES:
        backup = backup_dir / name
        target = root / name
        if backup.is_file():
            found.append((backup, target))
            print(f"[{ 'restore' if args.restore else 'would restore' }] {name}")
    if not found:
        print('No local source backups found. Nothing to restore.')
        return
    if not args.restore:
        print('Check only. Run with --restore to overwrite current local sources with backups.')
        return
    for backup, target in found:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup, target)
    print(f'[done] Restored {len(found)} local source files. No git operations executed.')


if __name__ == '__main__':
    main()
