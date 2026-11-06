"""Prepare reproducible held-out clip selections and official inference requests."""
import json
from pathlib import Path
import numpy as np
from .geometry import check_pose_array, check_actions, decode_actions, rotation_geodesic_deg


def read_jsonl(path):
    p=Path(path)
    if not p.is_file(): raise FileNotFoundError(p)
    rows=[]
    for i, line in enumerate(p.read_text(encoding='utf-8').splitlines(),1):
        if line.strip():
            try: rows.append(json.loads(line))
            except Exception as e: raise ValueError(f'Bad JSON line {i}: {p}') from e
    return rows


def write_jsonl(path, rows):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')


def verify_row(r, *, pose_tol_m=.01, pose_tol_deg=.1):
    cid = r['clip_id']
    v=Path(r['video']).expanduser()
    meta=Path(r['meta']).expanduser()
    action=Path(r['action_file']).expanduser()
    for p in (v,meta,action):
        if not p.is_file(): raise FileNotFoundError(f'{cid}: {p}')
    m=json.loads(meta.read_text(encoding='utf-8'))
    if m['clip_id']!=cid or m['dataset']!=r['dataset'] or m['split']!=r['split']:
        raise ValueError(f'{cid}: index vs metadata mismatch')
    if m.get('pose_convention')!='backward_framewise':
        raise ValueError(f'{cid}: expected backward_framewise, got {m.get("pose_convention")}')
    if m.get('action_normalization')!='none':
        raise ValueError(f'{cid}: unrecognized action normalization {m.get("action_normalization")}')
    if m.get('action_encoding')!='translation3 + rot6d_column_based':
        raise ValueError(f'{cid}: unrecognized action encoding {m.get("action_encoding")}')
    if m.get('fps')!=10 or m.get('frame_count')!=61 or m.get('action_count')!=60:
        raise ValueError(f'{cid}: expected 10fps, 61 frames, 60 actions')
    gt=check_pose_array(np.load(meta.parent/'poses_local_ego.npy',allow_pickle=False))
    acts=check_actions(np.load(action,allow_pickle=False))
    reconstructed=decode_actions(acts)
    max_pos=float(np.max(np.linalg.norm(gt[:,:3,3]-reconstructed[:,:3,3],axis=-1)))
    max_rot=float(np.max(rotation_geodesic_deg(gt[:,:3,:3],reconstructed[:,:3,:3])))
    if max_pos>pose_tol_m or max_rot>pose_tol_deg:
        raise ValueError(f'{cid}: GT action/pose convention mismatch: {max_pos:.6f}m, {max_rot:.6f}deg')
    timestamps=np.load(meta.parent/'timestamps_s.npy',allow_pickle=False)
    if timestamps.shape!=(61,) or not np.isfinite(timestamps).all() or np.min(np.diff(timestamps))<=0:
        raise ValueError(f'{cid}: invalid timestamps')
    return dict(gt_pose=gt,gt_actions=acts,metadata=m,timestamps=timestamps,
                max_gt_roundtrip_m=max_pos,max_gt_roundtrip_deg=max_rot)


def make_request(r, seed=0):
    return dict(name=r['clip_id'],model_mode='inverse_dynamics',domain_name='av',
                view_point='ego_view',vision_path=str(Path(r['video']).resolve()),
                action_chunk_size=60, fps=10, image_size=480,seed=seed,
                prompt='You are an autonomous vehicle planning system.')


def prepare(processed_root, out, datasets, split='val', n_per_dataset=3,seed=0,force=False):
    root=Path(processed_root)
    rows=read_jsonl(root/'index.jsonl')
    out=Path(out).expanduser().resolve()
    if out.exists() and (out/'selection.jsonl').exists() and not force:
        raise FileExistsError(f'{out}/selection.jsonl exists. Use --force to replace selection.')
    picked=[]
    for ds in datasets:
        group=sorted((r for r in rows if r['dataset']==ds and r['split']==split),key=lambda r:r['clip_id'])
        if not group: raise ValueError(f'No {split} clips found for {ds}; check processed/index.jsonl')
        for r in group[:n_per_dataset if n_per_dataset>0 else None]:
            verify_row(r)
            picked.append(r)
        print(f'[prepare] {ds}/{split}: {min(len(group),n_per_dataset) if n_per_dataset>0 else len(group)} clips')
    ids=[r['clip_id'] for r in picked]
    if len(set(ids))!=len(ids): raise ValueError('Duplicate clip IDs')
    out.mkdir(parents=True,exist_ok=True)
    write_jsonl(out/'selection.jsonl', picked)
    write_jsonl(out/'inference_requests.jsonl',[make_request(r,seed) for r in picked])
    print(f'[prepare OK] {len(picked)} clips. requests={out/"inference_requests.jsonl"}')
    return picked
