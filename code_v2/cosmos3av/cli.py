"""Stage-aware Cosmos3-Edge AV ID dataset and launch-plan generator."""
from __future__ import annotations
import argparse
import json
import os
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from .data import load_index, preflight, select_stage, mix_rows, write_jsonl


def _weights(value: str, selected: list[str]):
    if not value:
        return {k: 1. for k in selected}
    res = {}
    for item in value.split(','):
        if ':' not in item:
            raise ValueError('--weights expects kitti:1,waymo:1')
        k, v = item.split(':',1)
        if k in res:
            raise ValueError(f'Duplicate weight: {k}')
        res[k] = float(v)
    if set(res) != set(selected):
        raise ValueError(f'Weights must cover exactly {selected}')
    return res


def _toml(stage: str, name: str, iters: int, lr: float, save_every: int, gpus: int) -> str:
    # The official Edge action-SFT combination is unvalidated; start with few steps.
    return f'''# EXPERIMENTAL AV inverse dynamics adaptation of official Cosmos3 action SFT.
[job]
task = "vfm"
experiment = "action_inverse_dynamics_av_edge"
project = "cosmos3_av_id"
group = "{stage}"
name = "{name}"
wandb_mode = "disabled"

[model]
precision = "bfloat16"
max_num_tokens_after_packing = 45056
[model.parallelism]
data_parallel_shard_degree = {gpus}
data_parallel_replicate_degree = 1
[model.activation_checkpointing]
mode = "full"
save_ops_regex = ["fmha"]
[model.compile]
enabled = false  # re-enable once the first GPU training steps work

[model.tokenizer]
vae_path = "${{oc.env:WAN_VAE_PATH}}"

[optimizer]
lr = {lr}

[scheduler]
cycle_lengths = [{max(100, iters*2)}]
warm_up_steps = [{min(100, max(1, iters//10))}]

[trainer]
max_iter = {iters}
logging_iter = 10
grad_accum_iter = 1

[checkpoint]
load_path = "${{oc.env:BASE_CHECKPOINT_PATH}}"
save_iter = {save_every}
'''


def plan(args):
    root = Path(args.processed_root).expanduser().resolve()
    rows = load_index(root)
    selected = args.datasets or (['kitti', 'waymo'] if args.stage == 'pretrain' else ['av2'])
    selected = list(selected)
    train_rows = select_stage(rows, args.stage, selected, 'train')
    # Note validation is split-based. A dataset missing val does not block planning.
    val_rows = [r for r in rows if r['dataset'] in selected and r['split'] == 'val']
    weights = _weights(args.weights, selected)
    run_dir = Path(args.outdir).expanduser().resolve()
    if run_dir.exists() and (run_dir / 'run.json').exists() and not args.overwrite_plan:
        raise FileExistsError(f'Plan already exists: {run_dir}. Use a distinct --outdir or --overwrite-plan.')
    ckpt = Path(args.init_checkpoint).expanduser().resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f'Expected initialized Cosmos3 DCP checkpoint: {ckpt}')
    vae = Path(args.wan_vae).expanduser().resolve()
    if not vae.is_file():
        raise FileNotFoundError(f'Wan VAE checkpoint missing: {vae}')
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f'[source] available {dict(Counter((r["dataset"],r["split"]) for r in rows))}')
    print(f'[stage] {args.stage}, selected={selected}, weights={weights}')
    for ds in selected:
        print(f'[train] {ds}: {sum(r["dataset"] == ds for r in train_rows)} unique clips')
    subset = train_rows if args.full_preflight else train_rows[:min(50, len(train_rows))]
    worst = 0.0
    for r in subset:
        q = preflight(r, max_dt_error_s=args.max_dt_error_s, check_video=args.check_video)
        worst = max(worst, q['max_dt_error_s'])
    print(f'[preflight] checked={len(subset)} worst_dt_error_s={worst:.4f}')
    samples = mix_rows(train_rows, weights, args.epoch_clips, args.seed)
    for r in val_rows[:min(20, len(val_rows))]:
        preflight(r, max_dt_error_s=args.max_dt_error_s, check_video=False)
    manifest = run_dir / 'train.jsonl'
    write_jsonl(manifest, samples)
    write_jsonl(run_dir / 'val.jsonl', val_rows)
    (run_dir / 'sft_config.toml').write_text(_toml(args.stage, run_dir.name, args.max_iters,
                args.lr, args.save_every, args.gpus), encoding='utf8')
    detail = dict(stage=args.stage, datasets=selected, weights=weights,
        init_checkpoint=str(ckpt), wan_vae=str(vae), processed_root=str(root),
        train_manifest=str(manifest), val_manifest=str(run_dir/'val.jsonl'),
        n_train=len(samples), n_val=len(val_rows), gpus=args.gpus,
        max_dt_error_s=args.max_dt_error_s,
        caveat='EXPERIMENTAL: AV ID is not an NVIDIA validated SFT recipe; checkpoint action norm and head compatibility must be tested.',
    )
    (run_dir / 'run.json').write_text(json.dumps(detail, indent=2), encoding='utf8')
    print(f'[ready] {run_dir} with {len(samples)} train sampled entries and {len(val_rows)} val entries')
    print('Launch (from cosmos-framework root with this project on PYTHONPATH):')
    print(f"  python -m cosmos3av.cli launch --run-dir {shlex.quote(str(run_dir))} --framework-root /path/to/cosmos-framework")


def launch(args):
    rd = Path(args.run_dir).expanduser().resolve()
    detail = json.loads((rd / 'run.json').read_text(encoding='utf8'))
    framework = Path(args.framework_root).expanduser().resolve()
    if not (framework / 'cosmos_framework' / 'scripts' / 'train.py').is_file():
        raise FileNotFoundError(f'Not a Cosmos Framework checkout: {framework}')
    repo = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(repo), str(framework), env.get('PYTHONPATH', '')])
    env['COSMOS_AV_MANIFEST'] = detail['train_manifest']
    env['BASE_CHECKPOINT_PATH'] = detail['init_checkpoint']
    env['WAN_VAE_PATH'] = detail['wan_vae']
    env.setdefault('IMAGINAIRE_OUTPUT_ROOT', str(rd / 'outputs'))
    cmd = ['torchrun', '--standalone', f'--nproc_per_node={detail["gpus"]}',
           '-m', 'cosmos3av.entry', f'--sft-toml={rd / "sft_config.toml"}']
    print(' '.join(shlex.quote(x) for x in cmd))
    print('Manifest:', detail['train_manifest'])
    print('Checkpoint:', detail['init_checkpoint'])
    if not args.execute:
        print('[DRY RUN] Pass --execute to launch GPU training. This recipe is EXPERIMENTAL.')
        return
    subprocess.run(cmd, cwd=framework, env=env, check=True)


def stats(args):
    rows=load_index(Path(args.processed_root))
    chosen=[r for r in rows if args.dataset in ('all',r['dataset'])]
    if not chosen: raise ValueError('No clips')
    ds_stats={}
    for ds in sorted(set(r['dataset'] for r in chosen)):
        group=[r for r in chosen if r['dataset']==ds]
        errors=[]; durations=[]
        for r in group[:args.max_clips if args.max_clips else None]:
            q=preflight(r,max_dt_error_s=args.max_dt_error_s)
            errors.append(q['max_dt_error_s'])
            durations.append(q['mean_dt_s']*60)
        ds_stats[ds]={'clips_checked':len(errors),'mean_duration_s':float(np.mean(durations)),
          'max_abs_dt_error_s':float(np.max(errors)), 'nominal_fps':10}
    print(json.dumps(ds_stats,indent=2,ensure_ascii=False))


def smoke(args):
    """One-sample check against installed official Cosmos Framework action transform.

    Verifies Python/tensor contract only. It does NOT validate checkpoint action
    normalization or execute any GPU forward/backward pass.
    """
    from .framework_dataset import get_action_av_sft_dataset
    ds = get_action_av_sft_dataset(
        manifest=args.manifest, tokenizer_config=None, resolution=args.resolution,
        max_action_dim=64, iterable_shuffle=False,
    )
    sample = ds[0]
    print('[smoke] transformed sample keys:', sorted(sample.keys()))
    print('[smoke] video:', tuple(sample['video'].shape), sample['video'].dtype)
    print('[smoke] action:', tuple(sample['action'].shape), sample['action'].dtype)
    print('[smoke] sequence_plan:', sample.get('sequence_plan'))
    assert sample['video'].shape[1] == 61, 'Video length changed unexpectedly'
    assert sample['action'].shape[0] == 60, 'Action length changed unexpectedly'
    assert sample['action'].shape[-1] == 64, 'Expected padded 64D model action'
    assert 'sequence_plan' in sample, 'Missing Cosmos3 inverse dynamics sequence plan'
    print('[smoke OK] Action dataset and official transform contract; not a GPU training validation')


def main(argv=None):
    p=argparse.ArgumentParser(description='Cosmos3-Edge AV inverse dynamics SFT: plan, launch, timing QC')
    sub=p.add_subparsers(dest='cmd',required=True)
    q=sub.add_parser('plan', help='validate data and prepare stage-specific training manifests')
    q.add_argument('--processed-root',required=True)
    q.add_argument('--stage',choices=['pretrain','posttrain'],required=True)
    q.add_argument('--datasets',nargs='+', choices=['kitti','waymo','av2'])
    q.add_argument('--weights',default='',help='e.g. kitti:1,waymo:1')
    q.add_argument('--init-checkpoint',required=True,help='DCP checkpoint: Cosmos3-Edge for stage1, stage1 saved DCP for stage2')
    q.add_argument('--wan-vae',required=True)
    q.add_argument('--outdir',required=True)
    q.add_argument('--epoch-clips',type=int,default=1000)
    q.add_argument('--max-iters',type=int,default=100)
    q.add_argument('--save-every',type=int,default=50)
    q.add_argument('--lr',type=float,default=1e-5)
    q.add_argument('--gpus',type=int,default=8)
    q.add_argument('--seed',type=int,default=42)
    q.add_argument('--max-dt-error-s',type=float,default=.07)
    q.add_argument('--full-preflight',action='store_true')
    q.add_argument('--check-video',action='store_true')
    q.add_argument('--overwrite-plan',action='store_true')
    q.set_defaults(func=plan)
    x=sub.add_parser('launch',help='inspect command; use --execute for real experimental training')
    x.add_argument('--run-dir',required=True)
    x.add_argument('--framework-root',required=True)
    x.add_argument('--execute',action='store_true')
    x.set_defaults(func=launch)
    s=sub.add_parser('timing',help='inspect acquisition dt vs nominal 10Hz for any dataset')
    s.add_argument('--processed-root',required=True)
    s.add_argument('--dataset',choices=['all','kitti','waymo','av2'],default='all')
    s.add_argument('--max-clips',type=int,default=100)
    s.add_argument('--max-dt-error-s',type=float,default=.07)
    s.set_defaults(func=stats)
    t=sub.add_parser('smoke',help='try official Cosmos Framework action transform on one clip (no GPU)')
    t.add_argument('--manifest',required=True,help='generated run/train.jsonl')
    t.add_argument('--resolution',default='480')
    t.set_defaults(func=smoke)
    a=p.parse_args(argv)
    if a.cmd=='plan' and (a.max_iters < 1 or a.save_every < 1 or a.gpus < 1):
        p.error('--max-iters, --save-every and --gpus must be positive')
    return a.func(a)

if __name__=='__main__':
    main()
