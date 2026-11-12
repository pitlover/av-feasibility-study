# STAGE1_MULTISOURCE_FPS_V1 (local patch; no remote writes)
"""Held-out evaluation selections with V2/V4 metadata compatibility."""
import json
from pathlib import Path
import numpy as np
from .geometry import check_pose_array,check_actions,decode_actions,rotation_geodesic_deg
from cosmos3prep.contracts import V2,action_frame as resolve_action_frame,action_pose_path,normalized_metadata


def read_jsonl(path):
    path=Path(path)
    if not path.is_file():raise FileNotFoundError(path)
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def write_jsonl(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')


def verify_row(r,*,pose_tol_m=.01,pose_tol_deg=.1):
    cid=r['clip_id'];meta=Path(r['meta']);action=Path(r['action_file']);video=Path(r['video'])
    for f in (meta,action,video):
        if not f.is_file():raise FileNotFoundError(f'{cid}: missing {f}')
    m=json.loads(meta.read_text(encoding='utf-8'))
    for k in ('clip_id','dataset','split'):
        if m[k]!=r[k]:raise ValueError(f'{cid}: index vs metadata mismatch {k}')
    if m.get('pose_convention')!='backward_framewise':raise ValueError(f'{cid}: unsupported pose convention')
    if m.get('action_normalization')!='none':raise ValueError(f'{cid}: requires raw GT')
    enc=m.get('action_encoding')
    if m.get('format')==V2:
        if enc not in (None,'translation3 + rot6d_column_based'):
            raise ValueError(f'{cid}: unsupported V2 action encoding {enc}')
    elif enc!='translation3 + rot6d_column_based':
        raise ValueError(f'{cid}: unknown V4 action encoding {enc}')
    fps=m.get('fps')
    if type(fps) is not int or fps not in (10,12,20) or m.get('frame_count')!=61 or m.get('action_count')!=60:
        raise ValueError(f'{cid}: expected 10/12/20Hz, 61 frames and 60 actions')
    gt=check_pose_array(np.load(action_pose_path(m,meta.parent),allow_pickle=False))
    actions=check_actions(np.load(action,allow_pickle=False))
    recon=decode_actions(actions,m['pose_convention'])
    pos=float(np.max(np.linalg.norm(gt[:,:3,3]-recon[:,:3,3],axis=-1)))
    rot=float(np.max(rotation_geodesic_deg(gt[:,:3,:3],recon[:,:3,:3])))
    if pos>pose_tol_m or rot>pose_tol_deg:
        raise ValueError(f'{cid}: GT roundtrip mismatch {pos:.6f}m, {rot:.6f}deg')
    ts=np.load(meta.parent/'timestamps_s.npy',allow_pickle=False)
    if ts.shape!=(61,) or not np.isfinite(ts).all() or np.min(np.diff(ts))<=0:
        raise ValueError(f'{cid}: invalid timestamps')
    dt_error=float(np.max(np.abs(np.diff(ts)-1.0/fps)))
    tol={10:.07,12:.04,20:.025}[fps]
    if dt_error>tol:
        raise ValueError(f'{cid}: timestamp gaps exceed tolerance for {fps}Hz ({dt_error:.4f}s > {tol:.4f}s)')
    return {'gt_pose':gt,'gt_actions':actions,'metadata':normalized_metadata(m),
            'timestamps':ts,'max_gt_roundtrip_m':pos,'max_gt_roundtrip_deg':rot}


def make_request(r,seed=0):
    return {'name':r['clip_id'],'model_mode':'inverse_dynamics','domain_name':'av',
            'view_point':'ego_view','vision_path':str(Path(r['video']).resolve()),
            'action_chunk_size':60,'fps':int(json.loads(Path(r['meta']).read_text(encoding='utf-8'))['fps']),'image_size':480,'seed':seed,
            'prompt':'You are an autonomous vehicle planning system.'}


def prepare(processed_root,out,datasets,split='val',n_per_dataset=3,seed=0,force=False):
    rows=read_jsonl(Path(processed_root)/'index.jsonl')
    out=Path(out).expanduser().resolve()
    if (out/'selection.jsonl').is_file() and not force:raise FileExistsError(out/'selection.jsonl')
    selected=[]
    for ds in datasets:
        candidates=sorted((r for r in rows if r['dataset']==ds and r['split']==split),key=lambda x:x['clip_id'])
        if not candidates:raise ValueError(f'Missing {ds}/{split} clips')
        for r in candidates[:n_per_dataset if n_per_dataset>0 else None]:
            verify_row(r);selected.append(r)
    if len({r['clip_id'] for r in selected})!=len(selected):raise ValueError('Duplicate clip ids')
    out.mkdir(parents=True,exist_ok=True)
    write_jsonl(out/'selection.jsonl',selected)
    write_jsonl(out/'inference_requests.jsonl',[make_request(r,seed) for r in selected])
    print(f'[prepare OK] {len(selected)} held-out clips -> {out}')
    return selected
