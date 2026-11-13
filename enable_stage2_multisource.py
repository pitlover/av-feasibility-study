#!/usr/bin/env python3
"""Local-only patch for Stage-2 multisource Cosmos3 AV; never touches GitHub.

Requires feat/prepare_nuscenes source snapshot with stage1_source_index.py.
Run from repository root. Backs up modified files before writing; no remote writes.
"""
from pathlib import Path
import argparse
import ast
import shutil
import subprocess
import sys
import time


def once(src: str, before: str, after: str, label: str) -> str:
    n = src.count(before)
    if n != 1:
        raise ValueError(f'{label}: expected one anchor, found {n}: {before[:95]!r}')
    return src.replace(before, after, 1)


def section(src: str, start: str, stop: str, replacement: str, label: str) -> str:
    if src.count(start) != 1 or src.count(stop) != 1:
        raise ValueError(f'{label}: start/stop marker not unique or missing')
    a = src.index(start)
    b = src.index(stop, a)
    return src[:a] + replacement + src[b:]


def patched_parser(s: str) -> str:
    additions = """    'data.datasets': ('DATASETS_JSON', list, ['av2']),
    'data.sources.kitti': ('SOURCE_KITTI', str, ''),
    'data.sources.waymo': ('SOURCE_WAYMO', str, ''),
    'data.sources.av2': ('SOURCE_AV2', str, ''),
    'data.sources.nuscenes': ('SOURCE_NUSCENES', str, ''),
    'data.weights.kitti': ('WEIGHT_KITTI', float, 1.0),
    'data.weights.waymo': ('WEIGHT_WAYMO', float, 1.0),
    'data.weights.av2': ('WEIGHT_AV2', float, 1.0),
    'data.weights.nuscenes': ('WEIGHT_NUSCENES', float, 1.0),
"""
    s = once(s, "    'data.seed': ('SEED', int, 42),\n", "    'data.seed': ('SEED', int, 42),\n" + additions, 'stage2 parser fields')
    s = once(s, "    enabled, mode = values['wandb.enabled'], values['wandb.mode']", """    selected = values['data.datasets']
    allowed = {'kitti', 'waymo', 'av2', 'nuscenes'}
    if not selected or len(set(selected)) != len(selected) or set(selected) - allowed:
        raise ValueError(f'data.datasets must be a unique nonempty subset of {sorted(allowed)}')
    for ds in selected:
        if values[f'data.weights.{ds}'] <= 0:
            raise ValueError(f'data.weights.{ds} must be positive')
    enabled, mode = values['wandb.enabled'], values['wandb.mode']""", 'stage2 parser validate')
    s = once(s, "values['optimizer.lr'] <= 0 or values['eval.fps'] <= 0", "values['optimizer.lr'] <= 0 or values['eval.fps'] < 0", 'fps allow auto')
    s = once(s, "optimizer.lr and eval.fps must be positive", "optimizer.lr must be positive and eval.fps must be >=0 (0=per-clip auto)", 'fps error')
    s = once(s, "p.add_argument('--format', choices=('json','env0'), default='env0')", "p.add_argument('--format', choices=('json','env0','source-json'), default='env0')", 'parser format')
    s = once(s, """    if args.format == 'json':
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
""", """    if args.format == 'json':
        print(json.dumps(data, indent=2, ensure_ascii=False))
    elif args.format == 'source-json':
        selected = data['data.datasets']
        spec = {
            'base': data['paths.base'],
            'processed': data['paths.processed'],
            'datasets': selected,
            'sources': {ds: data[f'data.sources.{ds}'] for ds in selected},
            'weights': {ds: data[f'data.weights.{ds}'] for ds in selected},
        }
        print(json.dumps(spec, ensure_ascii=False))
    else:
""", 'parser output source-json')
    return s


def patched_data(s: str) -> str:
    return once(s, "    allowed = set(DATASETS) if stage == 'pretrain' else {'av2'}", """    if stage not in ('pretrain', 'posttrain'):
        raise ValueError(f'Unsupported stage: {stage}')
    allowed = set(DATASETS)""", 'posttrain allowed datasets')


def patched_shell(s: str) -> str:
    spec = r'''SPEC_CMD=(python "$HELPER" --format source-json)
[[ -z "$CONFIG_PATH" ]] || SPEC_CMD+=(--toml "$CONFIG_PATH")
SOURCE_SPEC_JSON="$("${SPEC_CMD[@]}" "${OVERRIDES[@]}")" || exit 2
DATASETS_JSON="$(python - "$SOURCE_SPEC_JSON" <<'PY_DS_JSON'
import json,sys
print(json.dumps(json.loads(sys.argv[1])['datasets']))
PY_DS_JSON
)"
DATASET_LIST=()
while IFS= read -r name; do
  [[ -n "$name" ]] && DATASET_LIST+=("$name")
done < <(python - "$DATASETS_JSON" <<'PY_DS_LIST'
import json,sys
for ds in json.loads(sys.argv[1]): print(ds)
PY_DS_LIST
)
DATASET_SUFFIX="$(IFS=_; echo "${DATASET_LIST[*]}")"
MIX_WEIGHTS="$(python - "$SOURCE_SPEC_JSON" <<'PY_MIX'
import json,sys
spec=json.loads(sys.argv[1]); print(','.join(f'{ds}:{spec["weights"][ds]}' for ds in spec['datasets']))
PY_MIX
)"
'''
    s = once(s, 'done\nif [[ "$MODE" == config ]]; then', 'done\n' + spec + 'if [[ "$MODE" == config ]]; then', 'stage2 source spec setup')
    s = once(s, 'EVAL="$BASE/evaluation/stage2_${RUN_TAG}/av2_${EVAL_TAG}"', 'EVAL="$BASE/evaluation/stage2_${RUN_TAG}/${DATASET_SUFFIX}_${EVAL_TAG}"', 'stage2 eval dir')
    s = once(s, "if info.get('action_frame') != 'camera':\n    raise SystemExit('[ERROR] Stage1 is not camera-relative; incompatible with current AV2 Stage2')", "if info.get('action_frame') != 'camera':\n    raise SystemExit('[ERROR] Stage1 is not camera-relative; incompatible with camera-action Stage2')", 'stage1 check message')
    s = once(s, '''case "$MODE" in
  plan|train|inspect|prepare|oracle|all) need_file "$PROCESSED/index.jsonl" ;;
  evaluate) [[ "$EVAL_MODE" != full ]] || need_file "$PROCESSED/index.jsonl" ;;
esac
''', '', 'remove single processed-root guard')
    s = once(s, "keys=('job.run_tag','paths.stage1_run','paths.stage1_checkpoint','paths.processed',\n      'data.epoch_clips','data.seed','train.gpus','train.max_iters',\n      'train.save_every','optimizer.lr','wandb.enabled','wandb.mode')\nbad=", "keys=('job.run_tag','paths.stage1_run','paths.stage1_checkpoint','paths.processed',\n      'data.datasets','data.epoch_clips','data.seed','train.gpus','train.max_iters',\n      'train.save_every','optimizer.lr','wandb.enabled','wandb.mode')\nkeys+=tuple(sorted(k for k in set(p)|set(n) if k.startswith(('data.sources.','data.weights.'))))\nbad=", 'audit config datasets/sources/weights')
    s = section(s, 'inspect(){\n', '\nplan(){\n', r'''inspect(){
  local tmp
  tmp="$(mktemp -d)"
  if ! python "$SCRIPT_DIR/scripts/stage1_source_index.py" --spec-json "$SOURCE_SPEC_JSON" --outdir "$tmp"; then
    rm -rf "$tmp"
    fail 'Stage2 source-index inspection failed'
  fi
  python - "$tmp/index.jsonl" "$DATASETS_JSON" <<'PY_INSPECT'
import json,sys
from collections import Counter
from pathlib import Path
rows=[json.loads(x) for x in Path(sys.argv[1]).read_text().splitlines() if x.strip()]
for ds in json.loads(sys.argv[2]):
    counts=Counter(r['split'] for r in rows if r['dataset']==ds)
    print(f'[STAGE2] {ds} clips={dict(counts)}')
    for split in ('train','val'):
        seq={r['sequence'] for r in rows if r['dataset']==ds and r['split']==split}
        print(f'  {split}: {len(seq)} sequences')
PY_INSPECT
  rm -rf "$tmp"
}
''', 'stage2 inspect')
    s = once(s, '''  python -m cosmos3av.cli plan --stage posttrain --datasets av2 \\
    --stage1-run "$STAGE1_RUN" \\
    --processed-root "$PROCESSED"''', '''  python "$SCRIPT_DIR/scripts/stage1_source_index.py" \\
    --spec-json "$SOURCE_SPEC_JSON" --outdir "$S2/input_sources" \\
    || fail 'Stage2 source index merge failed'
  python -m cosmos3av.cli plan --stage posttrain --datasets "${DATASET_LIST[@]}" \\
    --weights "$MIX_WEIGHTS" --stage1-run "$STAGE1_RUN" \\
    --processed-root "$S2/input_sources"''', 'stage2 plan multisource')
    s = once(s, "python - \"$S2/run.json\" \"$STAGE1_CKPT\" <<'PY'", "python - \"$S2/run.json\" \"$STAGE1_CKPT\" \"$DATASETS_JSON\" <<'PY'", 'plan check args')
    s = once(s, "assert d['stage']=='posttrain' and d['datasets']==['av2']", "assert d['stage']=='posttrain' and d['datasets']==json.loads(sys.argv[3])", 'plan check datasets')
    s = once(s, "echo '[TRAIN] AV2 posttraining from Stage1 DCP. Starting...'", "echo \"[TRAIN] Stage2 posttraining on ${DATASET_LIST[*]} from Stage1 DCP. Starting...\"", 'stage2 train message')
    s = section(s, 'prepare_eval(){\n', '\noracle(){\n', r'''prepare_eval(){
  if [[ -f "$EVAL/selection.jsonl" && -f "$EVAL/inference_requests.jsonl" ]]; then
    echo "[PREPARE] Using existing selection: $EVAL"
    return
  fi
  [[ ! -e "$EVAL" ]] || fail "Incomplete evaluation directory: $EVAL"
  local index_root="$S2/input_sources"
  need_file "$index_root/index.jsonl"
  python - "$index_root/index.jsonl" "$EVAL" "$EVAL_CLIPS" "$SEED" "$DATASETS_JSON" <<'PY_PREPARE'
import json,random,sys
from collections import defaultdict
from pathlib import Path
from cosmos3eval.data import read_jsonl,write_jsonl,verify_row,make_request
index,out=map(Path,sys.argv[1:3]); count=int(sys.argv[3]); seed=int(sys.argv[4]); datasets=json.loads(sys.argv[5])
rows=read_jsonl(index)
chosen=[]
for di,ds in enumerate(datasets):
    train={r['sequence'] for r in rows if r['dataset']==ds and r['split']=='train'}
    val=[r for r in rows if r['dataset']==ds and r['split']=='val']
    if not val: raise SystemExit(f'[ERROR] No {ds}/val clips found')
    overlap=train & {r['sequence'] for r in val}
    if overlap: raise SystemExit(f'[ERROR] {ds} train/val overlap: {sorted(overlap)[:5]}')
    groups=defaultdict(list)
    for r in val: groups[r['sequence']].append(r)
    rng=random.Random(seed+di); keys=sorted(groups); rng.shuffle(keys)
    for key in keys:rng.shuffle(groups[key])
    selected=[]
    while len(selected)<min(count,len(val)):
        any_added=False
        for key in keys:
            if groups[key]:
                selected.append(groups[key].pop()); any_added=True
                if len(selected)>=min(count,len(val)): break
        if not any_added: break
    chosen.extend(selected)
    print(f'[PREPARE] {ds}: val={len(selected)} clips / {len({r["sequence"] for r in selected})} sequences')
for r in chosen: verify_row(r)
out.mkdir(parents=True,exist_ok=False)
write_jsonl(out/'selection.jsonl',chosen)
write_jsonl(out/'inference_requests.jsonl',[make_request(r,seed) for r in chosen])
print(f'[PREPARE OK] selected {len(chosen)} clips from {datasets}')
PY_PREPARE
}
''', 'stage2 prepare_eval')
    s = once(s, "print('[STAGE2 AV2 EVALUATION]'", "print('[STAGE2 EVALUATION]'", 'stage2 score message')
    s = once(s, '''--selection "$EVAL/selection.jsonl" --datasets av2 \\
    --max-per-dataset''', '''--selection "$EVAL/selection.jsonl" --datasets "${DATASET_LIST[@]}" \\
    --max-per-dataset''', 'stage2 render selection')
    s = once(s, '''  local selected ds cid
  selected="$(python''', '''  local selected ds cid clip_fps
  selected="$(python''', 'stage2 render local')
    s = once(s, '''    echo "[RENDER] $ds clip=$cid (ground_height_m=$GROUND_HEIGHT_M approximate)"
    python -m cosmos3eval.cli render''', r'''    clip_fps="$(python - "$EVAL/selection.jsonl" "$cid" "$RENDER_FPS" <<'PY_RENDER_FPS'
import json,sys
from pathlib import Path
chosen=next(r for r in (json.loads(s) for s in Path(sys.argv[1]).read_text().splitlines() if s.strip()) if r['clip_id']==sys.argv[2])
requested=float(sys.argv[3]); meta=json.loads(Path(chosen['meta']).read_text(encoding='utf-8'))
print(requested if requested>0 else meta['fps'])
PY_RENDER_FPS
)"
    echo "[RENDER] $ds clip=$cid fps=$clip_fps (ground_height_m=$GROUND_HEIGHT_M approximate)"
    python -m cosmos3eval.cli render''', 'stage2 render dynamic fps')
    s = once(s, '''--fps "$RENDER_FPS" --out-dir "$PRED/qualitative/$ds"''', '''--fps "$clip_fps" --out-dir "$PRED/qualitative/$ds"''', 'stage2 render fps arg')
    return s


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project', default='/mnt/ufs/users/jaeha-lee/share/jiiyoung/av-feasibility-study')
    ap.add_argument('--apply',action='store_true',help='Apply local-only changes; defaults to read-only compatibility check')
    args=ap.parse_args()
    root=Path(args.project).expanduser().resolve()
    needed={
        'scripts/stage2_toml_config.py':patched_parser,
        'cosmos3av/data.py':patched_data,
        'stage2_full_train_eval.sh':patched_shell,
    }
    if not (root/'scripts/stage1_source_index.py').is_file():
        raise SystemExit('Missing scripts/stage1_source_index.py; Stage-1 multisource patch required')
    modified={}
    for name,fn in needed.items():
        p=root/name
        original=p.read_text(encoding='utf-8')
        new=fn(original)
        if new==original: raise RuntimeError(f'No modifications: {name}')
        modified[name]=(p,original,new)
        print('[PATCH READY]',name)
    # Statically validate generated Python before any writes.
    for name,(_,_,new) in modified.items():
        if name.endswith('.py'):ast.parse(new,filename=name)
    # Bash syntax check via stdin; no execution.
    shell=modified['stage2_full_train_eval.sh'][2]
    check=subprocess.run(['bash','-n'],input=shell,text=True,capture_output=True)
    if check.returncode:
        raise SystemExit('Bash syntax check failed:\n'+check.stderr)
    if not args.apply:
        print('[READ ONLY] Dry run successful. Pass --apply to patch local files after backups.')
        return
    backup=root/(f'.stage2_multisource_backup_{time.strftime("%Y%m%d_%H%M%S")}')
    backup.mkdir(exist_ok=False)
    for name,(p,original,new) in modified.items():
        dest=backup/name;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(original,encoding='utf-8')
    try:
        for name,(p,original,new) in modified.items():
            p.write_text(new,encoding='utf-8')
            print('[PATCHED LOCAL]',p)
    except Exception:
        for name,(p,original,_) in modified.items():
            p.write_text(original,encoding='utf-8')
        raise
    print('[DONE] Local-only change, no git commit/push. Backup:',backup)
    print('[NEXT] Create new config TOML; run config then inspect before planning.')

if __name__=='__main__':main()
