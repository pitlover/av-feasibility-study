"""Per-step signed camera Y rotational drift and trajectory translation diagnostics."""
import csv
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .data import read_jsonl,verify_row
from .predictions import load_pred
from .geometry import decode_actions,metrics


def diagnose(eval_dir,pred_dir,out,clip_id=None):
    selected=read_jsonl(Path(eval_dir)/'selection.jsonl')
    if clip_id:selected=[r for r in selected if r['clip_id']==clip_id]
    if not selected:raise ValueError('No matching clips')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    files=[]
    for row in selected:
        gt=verify_row(row);actions,_=load_pred(pred_dir,row['clip_id'])
        pred=decode_actions(actions,gt['metadata']['pose_convention'])
        target=gt['gt_pose']
        g=np.linalg.inv(target[:-1])@target[1:]
        p=np.linalg.inv(pred[:-1])@pred[1:]
        gy=Rotation.from_matrix(g[:,:3,:3]).as_rotvec()[:,1]
        py=Rotation.from_matrix(p[:,:3,:3]).as_rotvec()[:,1]
        m=metrics(target,pred)
        path=out/(row['clip_id']+'_steps.csv')
        fields=['step','gt_rotvec_y_deg','pred_rotvec_y_deg','signed_diff_deg',
                'gt_dx_m','pred_dx_m','gt_dz_m','pred_dz_m',
                'step_rre_deg','step_rte_m','cumulative_position_error_m','cumulative_rotation_error_deg']
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for i in range(60):
                writer.writerow(dict(step=i+1,gt_rotvec_y_deg=np.degrees(gy[i]),
                    pred_rotvec_y_deg=np.degrees(py[i]),signed_diff_deg=np.degrees(py[i]-gy[i]),
                    gt_dx_m=g[i,0,3],pred_dx_m=p[i,0,3],gt_dz_m=g[i,2,3],pred_dz_m=p[i,2,3],
                    step_rre_deg=m['rre_deg'][i],step_rte_m=m['rte_m'][i],
                    cumulative_position_error_m=m['ate_m'][i+1],
                    cumulative_rotation_error_deg=m['abs_rot_deg'][i+1]))
        files.append(path)
    return files
