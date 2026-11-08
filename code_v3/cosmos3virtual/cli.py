"""CLI: python -m cosmos3virtual.cli {kitti,waymo,av2,validate,fit-normalizer,denormalize-pred}."""
from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np

from cosmos3prep.datasets import iter_kitti,iter_waymo,iter_av2
from cosmos3prep.geometry import decode_local,validate_pose_array
from .camera import VirtualSpec
from .build import process_log,rebuild_virtual_index
from .normalization import fit_train_maxabs,read_norm,inverse_action


def build(args):
    spec=VirtualSpec(args.width,args.height,args.hfov)
    if args.max_source_gap_ms <= 0:raise ValueError('--max-source-gap-ms must be positive')
    # Native is the DEFAULT: use actual measured RGB timestamps and their matched GT.
    # Exact requires explicit opt-in and may create synthetic RGB exposures.
    out=Path(args.out).expanduser().resolve()
    source=Path(args.root).expanduser().resolve()
    if out == source or source in out.parents:
        raise ValueError('Refusing to write output inside the original dataset root')
    existing_index=out/'index.jsonl'
    if existing_index.is_file():
        old_rows=[json.loads(x) for x in existing_index.read_text().splitlines() if x.strip()]
        if old_rows:
            old_meta=Path(old_rows[0]['meta'])
            if old_meta.is_file():
                old_format=json.loads(old_meta.read_text()).get('format')
                if old_format!='cosmos3_virtual_camera_v2':
                    raise ValueError(f'Output root already contains non-virtual clips ({old_format}); choose a NEW --out directory')
    if args.dataset=='kitti':
        it=iter_kitti(Path(args.root),camera=args.camera, sequences=args.sequences,
                      poses_dir=Path(args.poses_dir) if args.poses_dir else None,
                      calib_root=Path(args.calib_root) if args.calib_root else None)
        if args.max_logs:
            import itertools
            it=itertools.islice(it,args.max_logs)
    elif args.dataset=='waymo':
        it=iter_waymo(Path(args.root),camera=args.camera,max_logs=args.max_logs,
                      compression=args.compression)
    else:
        it=iter_av2(Path(args.root),camera=args.camera,max_logs=args.max_logs)
    n=0
    for seq,split,frames in it:
        remaining=args.max_clips-n if args.max_clips else 10**12
        n+=process_log(frames,dataset=args.dataset,sequence=seq,split=split,spec=spec,
                       args=args,remaining=remaining)
        if args.max_clips and n>=args.max_clips:break
    total=rebuild_virtual_index(out)
    print(f'[done] new={n} indexed_total={total} root={out}')
    if n==0:
        print('[hint] existing clips may have been skipped; otherwise inspect sampling/FOV/calibration')


def validate(args):
    root=Path(args.out)
    idx=root/'index.jsonl'
    if not idx.is_file():raise FileNotFoundError(idx)
    rows=[json.loads(s) for s in idx.read_text().splitlines() if s.strip()]
    if not rows: raise ValueError('Index empty')
    by_dataset={}
    checked=0
    for row in rows[:args.max_clips if args.max_clips else None]:
        p=Path(row['meta']);m=json.loads(p.read_text())
        folder=p.parent
        if m.get('pose_frame')!='virtual_camera' or m.get('action_normalization')!='none':
            raise ValueError(f'{row["clip_id"]}: expected raw camera-coordinate GT')
        K=np.load(folder/'K_virtual.npy')
        poses=np.load(folder/'poses_local_virtual_camera.npy')
        acts=np.load(row['action_file'])
        T=np.load(folder/'T_ego_virtual_camera.npy')
        ts=np.load(folder/'timestamps_s.npy')
        if poses.shape!=(61,4,4) or acts.shape!=(60,9) or T.shape!=(61,4,4) or ts.shape!=(61,) or K.shape!=(3,3):
            raise ValueError(f'{row["clip_id"]}: bad dimensions')
        validate_pose_array(poses)
        validate_pose_array(T)
        if not np.all(np.diff(ts)>0): raise ValueError('Nonpositive timestamp interval')
        if m.get('time_mode')=='exact':
            max_error=float(np.max(np.abs(np.diff(ts)-.1)))
            if max_error > 1e-5:raise ValueError(f'{row["clip_id"]}: not exact 10Hz, {max_error}')
            if not (folder/'source_time_brackets_s.npy').exists():raise ValueError('Missing temporal source audit')
        elif m.get('time_mode')=='native':
            if int(m.get('synthetic_rgb_frames',0)) != 0:
                raise ValueError(f'{row["clip_id"]}: native mode must never synthesize RGB frames')
            max_error=float(np.max(np.abs(np.diff(ts)-.1)))
            if max_error > args.max_dt_error_s:
                raise ValueError(f'{row["clip_id"]}: native frame interval exceeds tolerance: {max_error:.5f}s')
        else:
            raise ValueError(f'{row["clip_id"]}: unsupported timing mode {m.get("time_mode")}')
        rec=decode_local(acts,m['pose_convention'])
        err=float(np.linalg.norm(rec[:,:3,3]-poses[:,:3,3],axis=1).max())
        if err>.002: raise ValueError(f'{row["clip_id"]}: GT/action mismatch {err}')
        if not (folder/'video.mp4').is_file():raise FileNotFoundError(folder/'video.mp4')
        import cv2
        v=cv2.VideoCapture(str(folder/'video.mp4'))
        if not v.isOpened():raise ValueError('Cannot decode MP4')
        count=0
        while True:
            ok,_=v.read()
            if not ok:break
            count+=1
        fps=v.get(cv2.CAP_PROP_FPS)
        v.release()
        if count!=61 or abs(fps-10)>.01:raise ValueError(f'{row["clip_id"]}: MP4 {count}frames {fps}FPS')
        by_dataset.setdefault(row['dataset'],[]).append(K)
        checked+=1
    if len(by_dataset)>1:
        all_K=[k for group in by_dataset.values() for k in group]
        if not np.allclose(all_K,all_K[0],atol=1e-6):
            raise ValueError('Virtual K mismatch between datasets')
    print(f'[VALIDATED] checked={checked}, datasets={list(by_dataset)} K/common/GT roundtrip/video/timestamp audit OK')


def fit(args):
    rows=[json.loads(s) for s in (Path(args.processed_root)/'index.jsonl').read_text().splitlines() if s.strip()]
    norm=fit_train_maxabs(rows, args.datasets)
    dst=Path(args.out)
    if dst.exists() and not args.force:raise FileExistsError(f'{dst} exists; use --force intentionally')
    dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text(json.dumps(norm,indent=2))
    print(f'[NORMALIZER FIT] train_only datasets={args.datasets}, scale_xyz_m={norm["xyz_scale_m"]}, saved={dst}')
    print('[WARNING] This is an explicit experimental train-set normalizer, NOT verified NVIDIA pretrained Edge statistics.')


def denorm_preds(args):
    # Prediction-space cannot be determined from 60x9 shape. Require explicit
    # acknowledgement before interpreting model samples as normalized values.
    if not args.assume_normalized_output:
        raise ValueError('Pass --assume-normalized-output ONLY after verifying official inference '
                         'is returning the custom-normalized XYZ (not already denormalized).')
    if not args.stage_run:
        raise ValueError('--stage-run required to bind predictions to the training normalizer')
    run=json.loads((Path(args.stage_run)/'run.json').read_text())
    norm_path=run.get('action_normalizer')
    if not norm_path:
        raise ValueError('This stage trained without an action normalizer; do not denormalize prediction')
    expected_hash=run.get('action_normalizer_sha256')
    if not expected_hash or hashlib.sha256(Path(norm_path).read_bytes()).hexdigest()!=expected_hash:
        raise RuntimeError('Training normalizer checksum mismatch')
    norm=read_norm(norm_path)
    src=Path(args.pred_dir).resolve();dst=Path(args.out).resolve()
    if src==dst or src in dst.parents:
        raise ValueError('Output directory must not be identical to or under the prediction directory')
    files=list(src.glob('*/sample_outputs.json'))
    if not files:raise FileNotFoundError('No clip/sample_outputs.json found')
    total=0
    for p in files:
        payload=json.loads(p.read_text())
        if payload.get('_cosmos_virtual_prediction_space') == 'raw_denormalized':
            raise ValueError(f'Already denormalized: {p}')
        # Actual Cosmos3 serialization: outputs[0].content.action
        node=payload
        if 'outputs' in node:
            if len(node['outputs'])!=1:raise ValueError('Ambiguous multi-output JSON')
            node=node['outputs'][0]
        if 'content' in node: node=node['content']
        if 'action' not in node:raise ValueError(f'Missing action in {p}')
        a=np.asarray(node['action'],dtype=np.float64)
        if a.ndim==3 and a.shape[0]==1:a=a[0]
        if a.ndim!=2 or a.shape[0]!=60 or a.shape[1] not in (9,64):
            raise ValueError(f'Invalid action prediction {a.shape} in {p}')
        inv=inverse_action(a,norm)
        node['action']=inv.tolist()
        payload['_cosmos_virtual_prediction_space']='raw_denormalized'
        payload['_cosmos_virtual_normalizer']=str(Path(norm_path).resolve())
        payload['_cosmos_virtual_normalizer_sha256']=expected_hash
        payload['_cosmos_virtual_training_run']=str(Path(args.stage_run).resolve())
        outp=dst/p.parent.name/p.name
        outp.parent.mkdir(parents=True,exist_ok=True)
        outp.write_text(json.dumps(payload,ensure_ascii=False))
        total+=1
    print(f'[DENORMALIZED] clips={total}; output={dst}; use this as --pred-dir for score/render')


def main(argv=None):
    p=argparse.ArgumentParser(description='Common virtual camera and camera-relative GT for Cosmos3')
    subs=p.add_subparsers(dest='cmd',required=True)
    for ds in ('kitti','waymo','av2'):
        q=subs.add_parser(ds)
        q.set_defaults(func=build,dataset=ds)
        q.add_argument('--root',required=True)
        q.add_argument('--out',required=True)
        q.add_argument('--width',type=int,default=640)
        q.add_argument('--height',type=int,default=384)
        q.add_argument('--hfov',type=float,default=45.0)
        q.add_argument('--camera',default={'kitti':'image_2','waymo':'FRONT','av2':'stereo_front_left'}[ds])
        q.add_argument('--stride',type=int,default=60)
        q.add_argument('--tolerance-ms',type=float,default=(55.0 if ds=='kitti' else 35.0))
        q.add_argument('--max-logs',type=int,default=0)
        q.add_argument('--max-clips',type=int,default=0)
        q.add_argument('--convention',choices=['backward_framewise','backward_anchored'],default='backward_framewise')
        q.add_argument('--encoder',choices=['local','official'],default='local')
        q.add_argument('--min-valid-fraction',type=float,default=.97)
        q.add_argument('--crf',type=int,default=18)
        q.add_argument('--strict',action='store_true')
        q.add_argument('--overwrite',action='store_true')
        q.add_argument('--kitti-timing',choices=['native','grid'],default='native',help='Only used in legacy native mode')
        q.add_argument('--time-mode',choices=['exact','native'],default='native',
                       help='native=original observed RGB and matching GT timestamps (default); exact=opt-in synthetic RGB 100ms')
        q.add_argument('--rgb-interpolation',choices=['flow','blend'],default='flow',
                       help='For exact-mode missing source RGB exposures; synthetic frames are flagged')
        q.add_argument('--max-source-gap-ms',type=float,default=160.0,
                       help='Split source sequence on large missing-frame gaps')
        q.add_argument('--compression',default='',choices=['','GZIP'])
        if ds=='kitti':
            q.add_argument('--poses-dir',required=True)
            q.add_argument('--calib-root',required=True)
            q.add_argument('--sequences',nargs='+')
        else:
            q.add_argument('--poses-dir',default=None)
            q.add_argument('--calib-root',default=None)
            q.add_argument('--sequences',nargs='+',default=None)
    v=subs.add_parser('validate');v.add_argument('--out',required=True);v.add_argument('--max-clips',type=int,default=0);v.add_argument('--max-dt-error-s',type=float,default=.07);v.set_defaults(func=validate)
    n=subs.add_parser('fit-normalizer');n.add_argument('--processed-root',required=True);n.add_argument('--datasets',nargs='+',choices=['kitti','waymo','av2'],required=True);n.add_argument('--out',required=True);n.add_argument('--force',action='store_true');n.set_defaults(func=fit)
    d=subs.add_parser('denormalize-pred');d.add_argument('--pred-dir',required=True)
    d.add_argument('--stage-run',required=True,help='run directory with run.json and immutable action_normalizer.json')
    d.add_argument('--assume-normalized-output',action='store_true',
                   help='Required acknowledgement: verify Cosmos inference has NOT already inversed the model action')
    d.add_argument('--out',required=True);d.set_defaults(func=denorm_preds)
    a=p.parse_args(argv)
    return a.func(a)

if __name__=='__main__':main()
