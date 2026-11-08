"""Cosmos3-Edge AV evaluation CLI: prepare, export, infer, score, render, diagnose."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from .data import prepare,read_jsonl
from .report import score,compare
from cosmos3prep.contracts import action_frame,image_mode


def _framework_env(root):
    root=Path(root).expanduser().resolve()
    if not (root/'cosmos_framework').is_dir():raise FileNotFoundError(root)
    env=os.environ.copy()
    env['PYTHONPATH']=os.pathsep.join(filter(None,(str(root),str(root.parent),env.get('PYTHONPATH',''))))
    return root,env


def _call(cmd,framework,execute):
    import shlex
    print('[COMMAND]',shlex.join([str(x) for x in cmd]),flush=True)
    if not execute:
        print('[DRY RUN] Pass --execute to run')
        return
    root,env=_framework_env(framework)
    subprocess.run(cmd,cwd=root,env=env,check=True)


def export_cmd(a):
    ck=Path(a.dcp).expanduser().resolve();cfg=Path(a.config).expanduser().resolve()
    if not ck.is_dir():raise FileNotFoundError(ck)
    if not cfg.is_file():raise FileNotFoundError(cfg)
    root,_=_framework_env(a.framework_root)
    out=Path(a.out).expanduser().resolve()
    sidecar=out.parent/(out.name+'.export_config.yaml')
    prep=[sys.executable,'-m','cosmos3eval.export_compat','--input',str(cfg),'--output',str(sidecar)]
    _call(prep,root,a.execute)
    _call([sys.executable,'-m','cosmos_framework.scripts.export_model',
           '--checkpoint-path',str(ck),'--config-file',str(sidecar),'-o',str(out)],root,a.execute)


def infer_cmd(a):
    root,_=_framework_env(a.framework_root)
    req=Path(a.eval_dir).expanduser().resolve()/'inference_requests.jsonl'
    if not req.is_file() or not read_jsonl(req):raise FileNotFoundError(req)
    checkpoint=a.checkpoint
    if checkpoint not in ('Cosmos3-Edge','Cosmos3-Nano','Cosmos3-Super'):
        p=Path(checkpoint).expanduser().resolve()
        if not p.exists():raise FileNotFoundError(p)
        if (p/'.metadata').exists() or (p/'__0_0.distcp').exists() or list(p.glob('*.distcp')):
            raise ValueError('Cannot infer directly on DCP; first run export')
        checkpoint=str(p)
    if a.gpus<1:raise ValueError('Expected gpus>=1')
    base=[sys.executable,'-m','cosmos_framework.scripts.inference',
          '--parallelism-preset=latency','-i',str(req),'-o',str(Path(a.pred_dir).expanduser().resolve()),
          '--checkpoint-path',checkpoint,'--seed',str(a.seed),'--no-guardrails']
    if a.gpus>1:
        cmd=[sys.executable,'-m','torch.distributed.run','--standalone',
             f'--nproc_per_node={a.gpus}',*base[1:],'--dp-shard-size=1']
    else:cmd=base
    _call(cmd,root,a.execute)


def render_cmd(a):
    if a.ego_height_m is not None:
        if a.ground_height_m is not None and a.ground_height_m!=a.ego_height_m:
            raise ValueError('Conflicting --ego-height-m and --ground-height-m')
        a.ground_height_m=a.ego_height_m
    if a.out_dir is None:a.out_dir=str(Path(a.pred_dir)/'qualitative')
    mode=a.mode
    if mode=='auto':
        rows=read_jsonl(Path(a.eval_dir)/'selection.jsonl')
        if a.clip_id:rows=[r for r in rows if r['clip_id']==a.clip_id]
        if not rows:raise ValueError('No selected clips')
        signatures=set()
        for row in rows[:a.max_clips if a.max_clips else None]:
            meta=json.loads(Path(row['meta']).read_text())
            signatures.add((action_frame(meta),image_mode(meta)))
        if len(signatures)!=1:raise ValueError(f'Mixed render contracts: {signatures}')
        mode='camera' if signatures=={('camera','virtual')} else 'legacy'
    if a.layout and mode!='camera':raise ValueError('--layout requires camera-based virtual GT')
    if mode=='camera':
        from .camera_render import render_camera_selection
        return render_camera_selection(a.eval_dir,a.pred_dir,a.out_dir,
            clip_id=a.clip_id,max_clips=a.max_clips,
            ground_height_m=a.ground_height_m,max_depth_m=a.max_depth_m,
            fps=a.fps,crf=a.crf,layout=a.layout or 'both',horizon_steps=a.horizon_steps)
    from .render import render_selection
    return render_selection(a.eval_dir,a.pred_dir,a.out_dir,clip_id=a.clip_id,
                            max_clips=a.max_clips,fps=a.fps,crf=a.crf)


def main(argv=None):
    ap=argparse.ArgumentParser(description='Cosmos3-Edge AV Inverse Dynamics eval')
    sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare')
    p.add_argument('--processed-root',required=True);p.add_argument('--out',required=True)
    p.add_argument('--datasets',nargs='+',choices=['kitti','waymo','av2'],required=True)
    p.add_argument('--split',choices=['train','val','test'],default='val')
    p.add_argument('--max-clips-per-dataset',type=int,default=3)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--force',action='store_true')
    p.set_defaults(func=lambda a:prepare(a.processed_root,a.out,a.datasets,a.split,a.max_clips_per_dataset,a.seed,a.force))
    p=sub.add_parser('export')
    p.add_argument('--framework-root',required=True);p.add_argument('--dcp',required=True)
    p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--execute',action='store_true');p.set_defaults(func=export_cmd)
    p=sub.add_parser('infer')
    p.add_argument('--eval-dir',required=True);p.add_argument('--pred-dir',required=True)
    p.add_argument('--framework-root',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--gpus',type=int,default=1);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--execute',action='store_true');p.set_defaults(func=infer_cmd)
    p=sub.add_parser('score')
    p.add_argument('--eval-dir',required=True);p.add_argument('--pred-dir',required=True)
    p.add_argument('--out');p.add_argument('--model',default='stage1')
    p.add_argument('--allow-padded-output',action='store_true')
    p.add_argument('--no-plots',action='store_true');p.add_argument('--skip-missing',action='store_true')
    p.add_argument('--oracle',action='store_true')
    p.set_defaults(func=lambda a:score(a.eval_dir,a.pred_dir,a.out,a.model,a.allow_padded_output,
                                      not a.no_plots,a.oracle,a.skip_missing))
    p=sub.add_parser('compare')
    p.add_argument('--summaries',nargs='+',required=True);p.add_argument('--out',required=True)
    p.set_defaults(func=lambda a:compare(a.summaries,a.out))
    p=sub.add_parser('render')
    p.add_argument('--eval-dir',required=True);p.add_argument('--pred-dir',required=True)
    p.add_argument('--out-dir');p.add_argument('--clip-id')
    p.add_argument('--model',default='model')
    p.add_argument('--mode',choices=['auto','camera','legacy'],default='auto')
    p.add_argument('--layout',choices=['both','projected','bev'])
    p.add_argument('--horizon-steps',type=int,default=60)
    p.add_argument('--ground-height-m',type=float);p.add_argument('--ego-height-m',type=float)
    p.add_argument('--max-depth-m',type=float,default=50.)
    p.add_argument('--max-clips',type=int,default=0)
    p.add_argument('--fps',type=float,default=10.);p.add_argument('--crf',type=int,default=18)
    p.set_defaults(func=render_cmd)
    p=sub.add_parser('diagnose')
    p.add_argument('--eval-dir',required=True);p.add_argument('--pred-dir',required=True)
    p.add_argument('--out-dir',required=True);p.add_argument('--clip-id')
    def diagnose_cmd(a):
        from .diagnose import diagnose
        for path in diagnose(a.eval_dir,a.pred_dir,a.out_dir,a.clip_id):print('[DIAGNOSE]',path)
    p.set_defaults(func=diagnose_cmd)
    a=ap.parse_args(argv)
    if a.command=='render' and (a.fps<=0 or not 0<=a.crf<=51 or a.horizon_steps<1 or a.horizon_steps>60):
        ap.error('Invalid render settings')
    a.func(a)

if __name__=='__main__':main()
