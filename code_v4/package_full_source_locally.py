#!/usr/bin/env python3
"""Package a locally patched Cosmos source tree into a ZIP without a network call.

Run only after reviewing source for secrets/credentials. Omits runtime data,
VCS metadata, backups, checkpoints, generated results and compiled caches.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

CODE_DIRS = ('cosmos3prep', 'cosmos3av', 'cosmos3eval', 'cosmos3virtual', 'scripts', 'tests')
ROOT_FILES = ('README.md', 'README_V4.md', 'V2_SYNC.md', 'check_config_contract.py',
              'requirements-base.txt', 'requirements-eval.txt', 'requirements-planning.txt')
ALLOWED_EXT = {'.py', '.md', '.txt', '.sh', '.toml', '.yaml', '.yml', '.json'}
BLOCKED_PARTS = {'.git', '__pycache__', '.pytest_cache', '.v2_metadata_backup', 'outputs', 'processed',
                 'processed_v4', 'processed_virtual_native_v2', 'runs', 'runs_virtual',
                 'evaluation', 'exports', 'checkpoints', 'normalizers'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args()
    root=args.root.expanduser().resolve()
    out=args.out.expanduser().resolve()
    if not (root/'cosmos3prep'/'contracts.py').is_file():
        ap.error('Run apply_patch.py first; cosmos3prep/contracts.py is missing')
    if root in out.parents:
        ap.error('Save ZIP outside source root')
    files=[]
    for d in CODE_DIRS:
        p=root/d
        if p.is_dir():
            files.extend(p.rglob('*'))
    for n in ROOT_FILES:
        p=root/n
        if p.is_file():files.append(p)
    files=sorted({p for p in files if p.is_file() and p.suffix.lower() in ALLOWED_EXT
                  and not set(p.relative_to(root).parts).intersection(BLOCKED_PARTS)})
    out.parent.mkdir(parents=True,exist_ok=True)
    with ZipFile(out,'w',ZIP_DEFLATED,compresslevel=9) as z:
        for p in files:
            z.write(p,'av-feasibility-study/'+str(p.relative_to(root)))
    print(f'[LOCAL CODE ZIP] {out}: {len(files)} source files')
    print('[NOTICE] Review the code for secrets before sharing. No network or GitHub write performed.')


if __name__=='__main__':main()
