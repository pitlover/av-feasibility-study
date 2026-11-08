"""Quantitative inverse-dynamics metrics, aggregate tables and per-clip BEV plots."""
import csv
import json
from pathlib import Path
import numpy as np
from .data import read_jsonl,verify_row
from .geometry import decode_actions,metrics
from .predictions import load_pred


def _plot(gt, pred, title, path, frame="ego"):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path=Path(path)
    fig,ax=plt.subplots(figsize=(7,6))
    x, y = (0,2) if frame == 'virtual_camera' else (0,1)
    ax.plot(gt[:,x,3],gt[:,y,3],label='GT',linewidth=2)
    ax.plot(pred[:,x,3],pred[:,y,3],label='Cosmos prediction',linewidth=2)
    ax.scatter([0],[0],marker='o',s=35,label='Start')
    ax.set_xlabel('camera X right (m)' if frame == 'virtual_camera' else 'ego X forward (m)')
    ax.set_ylabel('camera Z forward (m)' if frame == 'virtual_camera' else 'ego Y left (m)')
    ax.set_title(title);ax.axis('equal');ax.grid(alpha=.3);ax.legend()
    fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)


def _csv(path, rows,fields):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader();w.writerows(rows)


def score(eval_dir,pred_dir,out=None,model='stage1',allow_padded=False,plots=True,oracle=False,skip_missing=False):
    eval_dir=Path(eval_dir);pred_dir=Path(pred_dir)
    out=Path(out) if out else pred_dir/'metrics'
    out.mkdir(parents=True,exist_ok=True)
    rows=read_jsonl(eval_dir/'selection.jsonl')
    if not rows: raise ValueError('Empty selection')
    clips=[];frames=[]; failures=[]
    (out/'trajectories').mkdir(exist_ok=True)
    for r in rows:
        cid=r['clip_id']
        try:
            info=verify_row(r)
            gt=info['gt_pose']
            if oracle:
                act=info['gt_actions']
                pred_source='ORACLE_GT_DO_NOT_REPORT_AS_MODEL'
            else:
                act, pred_path=load_pred(pred_dir,cid,allow_padded=allow_padded)
                pred_source=str(pred_path)
            pred=decode_actions(act,convention=info['metadata']['pose_convention'])
            ms=metrics(gt,pred)
            timestamps=info['timestamps']
            per=dict(model=model,clip_id=cid,dataset=r['dataset'],split=r['split'],
                     sequence=r['sequence'],pose_frame=info['metadata']['pose_frame'],
                     rre_deg_mean=float(ms['rre_deg'].mean()),
                     rte_m_mean=float(ms['rte_m'].mean()),
                     ate_m_rmse=float(np.sqrt(np.mean(ms['ate_m']**2))),
                     final_position_error_m=float(ms['ate_m'][-1]),
                     final_rotation_error_deg=float(ms['abs_rot_deg'][-1]),
                     acquisition_duration_s=float(timestamps[-1]-timestamps[0]),
                     pred_source=pred_source)
            clips.append(per)
            for i in range(61):
                frames.append(dict(model=model,clip_id=cid,dataset=r['dataset'],
                    frame=i,timestamp_s=float(timestamps[i]),ate_m=float(ms['ate_m'][i]),
                    rre_deg=(float(ms['rre_deg'][i-1]) if i else ''),
                    rte_m=(float(ms['rte_m'][i-1]) if i else '')))
            np.save(out/'trajectories'/f'{cid}_pred.npy',pred)
            if plots: _plot(gt,pred,f'{model} | {cid}',out/f'bev_{cid}.png',frame=info['metadata']['pose_frame'])
            print(f'[OK] {cid}: RRE={per["rre_deg_mean"]:.4f}deg, RTE={per["rte_m_mean"]:.4f}m, ATE={per["ate_m_rmse"]:.4f}m')
        except Exception as e:
            failures.append(dict(clip_id=cid,error=str(e)))
            print(f'[ERROR] {cid}: {e}')
            if not skip_missing:
                (out/'failures.json').write_text(json.dumps(failures,indent=2,ensure_ascii=False))
                raise
    if not clips: raise RuntimeError('No valid clips scored')
    fields=list(clips[0]);_csv(out/'per_clip.csv',clips,fields)
    _csv(out/'per_frame.csv',frames,list(frames[0]))
    rre=[fr['rre_deg'] for fr in frames if fr['rre_deg']!='']
    rte=[fr['rte_m'] for fr in frames if fr['rte_m']!='']
    ate=[fr['ate_m'] for fr in frames]
    overall=dict(model=model,n_clips=len(clips),n_failed=len(failures),
        evaluated_clip_ids=sorted(c['clip_id'] for c in clips),
        rre_deg_mean=float(np.mean(rre)),rte_m_mean=float(np.mean(rte)),
        ate_m_rmse=float(np.sqrt(np.mean(np.square(ate)))),
        final_position_error_m_mean=float(np.mean([x['final_position_error_m'] for x in clips])),
        metrics_definition='RRE/RTE: consecutive relative SE3 errors; ATE: local-origin position RMSE, no Sim3 alignment',
        oracle=oracle,representative_pose_frames=sorted(set(c['pose_frame'] for c in clips)))
    by_dataset={}
    for ds in sorted(set(c['dataset'] for c in clips)):
        cs=[c for c in clips if c['dataset']==ds]
        fs=[f for f in frames if f['dataset']==ds]
        by_dataset[ds]=dict(n_clips=len(cs),
            rre_deg_mean=float(np.mean([f['rre_deg'] for f in fs if f['rre_deg']!=''])),
            rte_m_mean=float(np.mean([f['rte_m'] for f in fs if f['rte_m']!=''])),
            ate_m_rmse=float(np.sqrt(np.mean([f['ate_m']**2 for f in fs]))))
    summary=dict(overall=overall,by_dataset=by_dataset,failures=failures)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print('[SUMMARY]',json.dumps(overall,indent=2,ensure_ascii=False))
    print('[FILES]',out)
    if failures:
        (out/'failures.json').write_text(json.dumps(failures,indent=2,ensure_ascii=False))
    return summary


def compare(files,out):
    rows=[]; reference_clips=None
    for p in files:
        j=json.loads(Path(p).read_text(encoding='utf8'))
        if j['overall'].get('oracle'):
            raise ValueError('Cannot compare GT oracle with real model metrics')
        these_clips=j['overall'].get('evaluated_clip_ids')
        if these_clips is None:raise ValueError('Summary missing evaluated_clip_ids')
        if reference_clips is None: reference_clips=these_clips
        elif these_clips != reference_clips:
            raise ValueError('Different evaluation clip sets. Re-score the exact same clips for every model.')
        rows.append({k:j['overall'][k] for k in ('model','n_clips','rre_deg_mean','rte_m_mean','ate_m_rmse')})
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    _csv(out,rows,list(rows[0]));print('[COMPARE]',out)
