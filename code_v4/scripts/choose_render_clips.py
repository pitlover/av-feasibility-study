#!/usr/bin/env python3
"""Choose validated clips from a prepared evaluation selection for rendering.

Emits <dataset>\t<clip_id> rows to stdout. Nothing is modified.
"""
import argparse
import json
import sys
from pathlib import Path


def select(selection, datasets, max_per_dataset, explicit):
    if max_per_dataset < 1:
        raise ValueError('max_per_dataset must be >= 1')
    rows = [json.loads(line) for line in Path(selection).read_text(encoding='utf8').splitlines() if line.strip()]
    available = {row['clip_id']: row for row in rows}
    if len(available) != len(rows):
        raise ValueError('Duplicate clip IDs in selection.jsonl')
    if not rows:
        raise ValueError('Empty evaluation selection')
    if explicit:
        missing = sorted(set(explicit)-set(available))
        if missing:
            raise ValueError('Selected clip IDs not in this evaluation selection: ' + ', '.join(missing))
        invalid = [cid for cid in explicit if available[cid]['dataset'] not in datasets]
        if invalid:
            raise ValueError('Clip IDs belong to a different dataset: '+', '.join(invalid))
        return [(available[cid]['dataset'],cid) for cid in explicit]
    selected=[]
    for ds in datasets:
        chosen=[r for r in rows if r['dataset']==ds][:max_per_dataset]
        if not chosen:
            raise ValueError(f'No {ds} clips in this evaluation selection')
        selected.extend((ds,r['clip_id']) for r in chosen)
    return selected


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--selection',required=True)
    ap.add_argument('--datasets',nargs='+',required=True)
    ap.add_argument('--max-per-dataset',type=int,default=1)
    ap.add_argument('--clip-ids-json',default='[]')
    args=ap.parse_args()
    try:
        explicit=json.loads(args.clip_ids_json)
        if not isinstance(explicit,list) or not all(isinstance(s,str) for s in explicit):
            raise ValueError('clip-ids-json must be a JSON array of clip ID strings')
        if len(explicit)!=len(set(explicit)):
            raise ValueError('Duplicate clip IDs')
        result=select(args.selection,args.datasets,args.max_per_dataset,explicit)
    except (OSError,ValueError,KeyError,json.JSONDecodeError) as exc:
        print(f'[RENDER ERROR] {exc}',file=sys.stderr)
        return 2
    for ds,cid in result:
        print(f'{ds}\t{cid}')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
