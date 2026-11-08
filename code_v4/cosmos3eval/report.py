"""Compute per-frame/clip RRE, RTE, ATE and save BEV plots."""
import csv
import json
from pathlib import Path
import numpy as np
from .data import read_jsonl,verify_row
from .geometry import decode_actions,metrics
from .predictions import load_pred


def _csv(path,rows,fields):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def _plot(gt,pred,title,path,action_frame='ego'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    forward,lateral=(2,0) if action_frame=='camera' else (0,1)
    fig,ax=plt.subplots(figsize=(7,6))
    ax.plot(gt[:,forward,3],gt[:,lateral,3],label='GT',linewidth=2)
    ax.plot(pred[:,forward,3],pred[:,lateral,3],label='Cosmos prediction',linewidth=2)
    ax.scatter([0],[0],s=35,label='Start');ax.axis('equal');ax.grid(alpha=.3);ax.legend()
    ax.set_xlabel('Forward Z (m)' if action_frame=='camera' else 'Vehicle forward X (m)')
    ax.set_ylabel('Camera X right (m)' if action_frame=='camera' else 'Vehicle Y left (m)')
    ax.set_title(title);fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)


def score(eval_dir,pred_dir,out=None,model='stage1',allow_padded=False,plots=True,oracle=False,skip_missing=False):
    eval_dir=Path(eval_dir);pred_dir=Path(pred_dir)
    out=Path(out) if out else pred_dir/'metrics'
    out.mkdir(parents=True,exist_ok=True)
    rows=read_jsonl(eval_dir/'selection.jsonl')
    if not rows:raise ValueError('Empty evaluation selection')
    clips=[];frames=[];failures=[]
    (out/'trajectories').mkdir(exist_ok=True)
    for row in rows:
        cid=row['clip_id']
        try:
            info=verify_row(row);gt=info['gt_pose']
            if oracle:
                actions=info['gt_actions'];source='ORACLE_GT_DO_NOT_REPORT_AS_MODEL'
            else:
                actions,source=load_pred(pred_dir,cid,allow_padded)
            pred=decode_actions(actions,info['metadata']['pose_convention'])
            m=metrics(gt,pred)
            ts=info['timestamps'];frame=info['metadata'].get('action_frame','ego')
            entry=dict(model=model,clip_id=cid,dataset=row['dataset'],split=row['split'],
                sequence=row['sequence'],pose_frame=frame,
                rre_deg_mean=float(m['rre_deg'].mean()),rte_m_mean=float(m['rte_m'].mean()),
                ate_m_rmse=float(np.sqrt(np.mean(m['ate_m']**2))),
                final_position_error_m=float(m['ate_m'][-1]),
                final_rotation_error_deg=float(m['abs_rot_deg'][-1]),
                acquisition_duration_s=float(ts[-1]-ts[0]),pred_source=str(source))
            clips.append(entry)
            for i in range(61):
                frames.append(dict(model=model,clip_id=cid,dataset=row['dataset'],frame=i,
                    timestamp_s=float(ts[i]),ate_m=float(m['ate_m'][i]),
                    rre_deg=float(m['rre_deg'][i-1]) if i else '',
                    rte_m=float(m['rte_m'][i-1]) if i else ''))
            np.save(out/'trajectories'/f'{cid}_pred.npy',pred)
            if plots:_plot(gt,pred,f'{model} | {cid}',out/f'bev_{cid}.png',frame)
            print(f'[SCORE OK] {cid}: RRE={entry["rre_deg_mean"]:.4f}, RTE={entry["rte_m_mean"]:.4f}, ATE={entry["ate_m_rmse"]:.4f}')
        except Exception as e:
            failures.append(dict(clip_id=cid,error=str(e)))
            if not skip_missing:
                (out/'failures.json').write_text(json.dumps(failures,indent=2));raise
    if not clips:raise RuntimeError('No valid clips scored')
    _csv(out/'per_clip.csv',clips,list(clips[0]))
    _csv(out/'per_frame.csv',frames,list(frames[0]))
    overall=dict(model=model,n_clips=len(clips),n_failed=len(failures),
        evaluated_clip_ids=sorted(c['clip_id'] for c in clips),
        rre_deg_mean=float(np.mean([f['rre_deg'] for f in frames if f['rre_deg']!=''])),
        rte_m_mean=float(np.mean([f['rte_m'] for f in frames if f['rte_m']!=''])),
        ate_m_rmse=float(np.sqrt(np.mean([f['ate_m']**2 for f in frames]))),
        final_position_error_m_mean=float(np.mean([c['final_position_error_m'] for c in clips])),
        metrics_definition='RRE/RTE: consecutive relative SE3, ATE: local-origin RMSE (no Sim3 alignment)',
        oracle=oracle,representative_pose_frames=sorted({c['pose_frame'] for c in clips}))
    by_dataset={}
    for ds in sorted({c['dataset'] for c in clips}):
        ff=[f for f in frames if f['dataset']==ds]
        by_dataset[ds]=dict(n_clips=sum(c['dataset']==ds for c in clips),
            rre_deg_mean=float(np.mean([f['rre_deg'] for f in ff if f['rre_deg']!=''])),
            rte_m_mean=float(np.mean([f['rte_m'] for f in ff if f['rte_m']!=''])),
            ate_m_rmse=float(np.sqrt(np.mean([f['ate_m']**2 for f in ff]))))
    summary=dict(overall=overall,by_dataset=by_dataset,failures=failures)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    if failures:(out/'failures.json').write_text(json.dumps(failures,indent=2))
    return summary


def compare(files,out):
    rows=[];expected=None
    for path in files:
        d=json.loads(Path(path).read_text());overall=d['overall']
        if overall.get('oracle'):raise ValueError('GT oracle is not a model result')
        clips=overall['evaluated_clip_ids']
        if expected is None:expected=clips
        elif expected!=clips:raise ValueError('Compare only matching evaluation clip IDs')
        rows.append({k:overall[k] for k in ('model','n_clips','rre_deg_mean','rte_m_mean','ate_m_rmse')})
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    _csv(out,rows,list(rows[0]));return out
