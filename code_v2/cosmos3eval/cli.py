"""CLI: prepare, export, infer, score, compare, selftest.

Evaluate NVIDIA Cosmos3-Edge AV inverse dynamics using existing cosmos3prep
processed/index.jsonl. Inference runs through the OFFICIAL cosmos_framework CLI;
all metric calculations are standalone and CPU-only.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .data import prepare,read_jsonl
from .report import score,compare


def _framework_env(root):
    root=Path(root).expanduser().resolve()
    if not (root/'cosmos_framework').is_dir():
        raise FileNotFoundError(f'--framework-root must contain cosmos_framework/: {root}')
    env=os.environ.copy()
    env['PYTHONPATH']=os.pathsep.join(filter(None,(str(root),str(root.parent),env.get('PYTHONPATH',''))))
    return root,env


def _call(cmd,framework,execute):
    import shlex
    print('[COMMAND]',shlex.join([str(x) for x in cmd]),flush=True)
    if not execute:
        print('[DRY RUN] Add --execute to run this command')
        return
    root,env=_framework_env(framework)
    subprocess.run(cmd,cwd=root,env=env,check=True)


def export_cmd(a):
    ckpt=Path(a.dcp).expanduser().resolve()
    config=Path(a.config).expanduser().resolve()
    if not ckpt.is_dir():raise FileNotFoundError(f'DCP directory: {ckpt}')
    if not config.is_file():raise FileNotFoundError(f'Training config.yaml: {config}')
    root,_=_framework_env(a.framework_root)
    cmd=[sys.executable,'-m','cosmos_framework.scripts.export_model',
         '--checkpoint-path',str(ckpt),'--config-file',str(config),
         '-o',str(Path(a.out).expanduser().resolve())]
    _call(cmd,root,a.execute)


def infer_cmd(a):
    root,_=_framework_env(a.framework_root)
    evaluation=Path(a.eval_dir).expanduser().resolve()
    req=evaluation/'inference_requests.jsonl'
    if not req.is_file():raise FileNotFoundError(req)
    r=read_jsonl(req)
    if not r:raise ValueError('No inference requests')
    out=Path(a.pred_dir).expanduser().resolve()
    checkpoint=a.checkpoint
    if checkpoint not in ('Cosmos3-Edge','Cosmos3-Nano','Cosmos3-Super'):
        p=Path(checkpoint).expanduser().resolve()
        if not p.exists():raise FileNotFoundError(f'Checkpoint not found: {p}')
        if (p/'.metadata').exists() or (p/'__0_0.distcp').exists() or list(p.glob('*.distcp')):
            raise ValueError('Raw DCP cannot be used directly for official inference. Use `export` first, then --checkpoint path/to/exported/model')
        checkpoint=str(p)
    if a.gpus<1:raise ValueError('--gpus must be >=1')
    base=[sys.executable,'-m','cosmos_framework.scripts.inference',
          '--parallelism-preset=latency','-i',str(req),'-o',str(out),
          '--checkpoint-path',checkpoint,'--seed',str(a.seed),'--no-guardrails']
    if a.gpus>1:
        cmd=[sys.executable,'-m','torch.distributed.run','--standalone',
             f'--nproc_per_node={a.gpus}',*base[1:],'--dp-shard-size=1']
    else:cmd=base
    _call(cmd,root,a.execute)


def prep_cmd(a):
    prepare(a.processed_root,a.out,a.datasets,a.split,a.max_clips_per_dataset,a.seed,a.force)


def score_cmd(a):
    score(a.eval_dir,a.pred_dir,a.out,a.model,a.allow_padded_output,
          not a.no_plots,a.oracle,a.skip_missing)


def compare_cmd(a):
    compare(a.summaries,a.out)


def main(argv=None):
    ap=argparse.ArgumentParser(description='Cosmos3-Edge AV inverse dynamics: inference + RRE/RTE/ATE')
    sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare',help='Create held-out selection and official inference JSONL (no GPU)')
    p.add_argument('--processed-root',required=True);p.add_argument('--out',required=True)
    p.add_argument('--datasets',nargs='+',choices=['kitti','waymo','av2'],required=True)
    p.add_argument('--split',choices=['train','val','test'],default='val')
    p.add_argument('--max-clips-per-dataset',type=int,default=3,help='0=all')
    p.add_argument('--seed',type=int,default=0);p.add_argument('--force',action='store_true')
    p.set_defaults(func=prep_cmd)
    e=sub.add_parser('export',help='Export training DCP to official Cosmos inference checkpoint')
    e.add_argument('--framework-root',required=True);e.add_argument('--dcp',required=True)
    e.add_argument('--config',required=True,help='training output config.yaml, NOT sft_config.toml')
    e.add_argument('--out',required=True);e.add_argument('--execute',action='store_true')
    e.set_defaults(func=export_cmd)
    inf=sub.add_parser('infer',help='Official Cosmos3 inference, requires exported checkpoint or model name')
    inf.add_argument('--eval-dir',required=True);inf.add_argument('--pred-dir',required=True)
    inf.add_argument('--framework-root',required=True);inf.add_argument('--checkpoint',required=True)
    inf.add_argument('--gpus',type=int,default=1);inf.add_argument('--seed',type=int,default=0)
    inf.add_argument('--execute',action='store_true');inf.set_defaults(func=infer_cmd)
    s=sub.add_parser('score',help='Score real model prediction JSON/NPY, write CSV/BEV/summary')
    s.add_argument('--eval-dir',required=True);s.add_argument('--pred-dir',required=True)
    s.add_argument('--out');s.add_argument('--model',default='stage1_kitti')
    s.add_argument('--allow-padded-output',action='store_true')
    s.add_argument('--no-plots',action='store_true');s.add_argument('--skip-missing',action='store_true')
    s.add_argument('--oracle',action='store_true',help='TEST ONLY: compare GT action to GT pose; NEVER a model result')
    s.set_defaults(func=score_cmd)
    c=sub.add_parser('compare',help='Combine score summaries from identically selected clips')
    c.add_argument('--summaries',nargs='+',required=True);c.add_argument('--out',required=True)
    c.set_defaults(func=compare_cmd)
    a=ap.parse_args(argv)
    if a.command=='prepare' and a.max_clips_per_dataset<0:ap.error('--max-clips-per-dataset must be >=0')
    a.func(a)

if __name__=='__main__':main()
