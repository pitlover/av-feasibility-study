"""Camera-optical future-road trajectory overlay and camera-frame BEV comparison.

Camera X right, Y down, Z forward. Ground projection assumes a local flat
road plane and approximated camera height. Do not project camera centers.
"""
import json
from pathlib import Path
import cv2
import numpy as np
from cosmos3virtual.camera import project_camera_xyz
from cosmos3prep.contracts import video_intrinsics
from .data import verify_row,read_jsonl
from .geometry import decode_actions,metrics
from .predictions import load_pred
from .render import _encode_h264

GT_COLOR=(65,210,85)
PRED_COLOR=(75,80,240)


def future_road_points(local_poses,frame_index,ground_height_m):
    T=np.asarray(local_poses,np.float64)
    origin=np.array([0.,ground_height_m,0.,1.])
    transform=np.linalg.inv(T[frame_index])[None]@T[frame_index:]
    return (transform@origin)[:,:3]


def _draw_road(frame,poses,i,K,color,*,ground_height_m,max_depth,horizon_steps=60):
    points=future_road_points(poses,i,ground_height_m)[:horizon_steps+1]
    uv,valid=project_camera_xyz(points,K,min_z=.8,max_z=max_depth)
    h,w=frame.shape[:2]
    valid&=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    segment=[]
    for pt,ok in zip(uv,valid):
        if ok:segment.append(tuple(np.rint(pt).astype(int)))
        else:
            if len(segment)>1:cv2.polylines(frame,[np.asarray(segment,np.int32)],False,color,3,cv2.LINE_AA)
            segment=[]
    if len(segment)>1:cv2.polylines(frame,[np.asarray(segment,np.int32)],False,color,3,cv2.LINE_AA)
    if np.any(valid):cv2.circle(frame,tuple(np.rint(uv[np.flatnonzero(valid)[0]]).astype(int)),4,color,-1,cv2.LINE_AA)
    return frame


def _bev(gt,pred,upto,size=360):
    canvas=np.full((size,size,3),247,np.uint8)
    pts=np.concatenate([gt[:,[0,2],3],pred[:,[0,2],3]])
    pts[:,1]*=-1
    lo=pts.min(axis=0);hi=pts.max(axis=0)
    sc=min((size-42)/e for e in np.maximum(hi-lo,1.))
    pix=np.rint((pts-(lo+hi)/2)*sc+size/2).astype(np.int32)
    n=len(gt)
    for path,col in ((pix[:upto+1],GT_COLOR),(pix[n:n+upto+1],PRED_COLOR)):
        if len(path)>1:cv2.polylines(canvas,[path.reshape(-1,1,2)],False,col,2,cv2.LINE_AA)
        cv2.circle(canvas,tuple(path[-1]),5,col,-1,cv2.LINE_AA)
    cv2.putText(canvas,'Camera BEV: X-right / Z-forward',(8,18),cv2.FONT_HERSHEY_SIMPLEX,.42,(25,25,25),1,cv2.LINE_AA)
    cv2.putText(canvas,'GT green | Pred red',(8,size-9),cv2.FONT_HERSHEY_SIMPLEX,.45,(25,25,25),1,cv2.LINE_AA)
    return canvas


def render_camera_clip(row,pred_dir,out_path,*,ground_height_m=None,max_depth_m=50.,fps=10.,crf=18,layout='both',horizon_steps=60):
    if layout not in ('both','projected','bev') or not 1<=horizon_steps<=60:
        raise ValueError('Invalid layout/horizon')
    info=verify_row(row);meta=info['metadata']
    if meta.get('action_frame')!='camera':raise ValueError('Camera render requires camera GT')
    if meta.get('image_mode')!='virtual':raise ValueError('Camera render requires virtual camera video')
    K=video_intrinsics(meta)
    if ground_height_m is None:
        ec=np.asarray(meta['T_ego_camera_first'],np.float64)
        ground_height_m=float(ec[2,3])
        if not .4<=ground_height_m<=4.:
            raise ValueError('Provide calibrated --ground-height-m or --ego-height-m')
    if ground_height_m<=0:raise ValueError('Ground height must be positive')
    a,_=load_pred(pred_dir,row['clip_id'])
    pred=decode_actions(a,meta['pose_convention']);gt=info['gt_pose']
    m=metrics(gt,pred)
    cap=cv2.VideoCapture(str(row['video']))
    if not cap.isOpened():raise OSError(row['video'])
    w,h=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    bev_size=min(h,280,max(80,int(h*.56)))
    out_w=w+bev_size if layout=='both' else w
    out_w+=out_w%2;out_h=h+(h%2)
    def frames():
        for i in range(61):
            ok,frame=cap.read()
            if not ok:raise ValueError('Input MP4 has fewer than 61 frames')
            if layout!='bev':
                _draw_road(frame,gt,i,K,GT_COLOR,ground_height_m=ground_height_m,max_depth=max_depth_m,horizon_steps=horizon_steps)
                _draw_road(frame,pred,i,K,PRED_COLOR,ground_height_m=ground_height_m,max_depth=max_depth_m,horizon_steps=horizon_steps)
            canvas=np.full((out_h,out_w,3),245,np.uint8)
            canvas[:h,:w]=frame
            bev=_bev(gt,pred,i,bev_size)
            if layout=='both':
                canvas[:bev_size,w:w+bev_size]=bev
                text=[f'Frame {i:02d}/60',f'Position error {m["ate_m"][i]:.3f} m',
                      f'Rotation error {m["abs_rot_deg"][i]:.3f} deg',
                      f'RRE {np.mean(m["rre_deg"][:max(i,1)]):.3f} deg',
                      f'RTE {np.mean(m["rte_m"][:max(i,1)]):.3f} m']
                for j,line in enumerate(text):
                    cv2.putText(canvas,line,(w+10,bev_size+25+j*25),cv2.FONT_HERSHEY_SIMPLEX,.47,(25,25,25),1,cv2.LINE_AA)
            elif layout=='bev':canvas[:bev_size,:bev_size]=bev
            yield canvas
    try:_encode_h264(frames(),out_path,fps,out_w,out_h,crf)
    finally:cap.release()
    return dict(clip_id=row['clip_id'],final_position_error_m=float(m['ate_m'][-1]),
                final_rotation_error_deg=float(m['abs_rot_deg'][-1]),
                mean_step_rre_deg=float(m['rre_deg'].mean()),
                mean_step_rte_m=float(m['rte_m'].mean()),
                gt_action_roundtrip_m=info['max_gt_roundtrip_m'],
                ground_height_m=ground_height_m,ground_plane_is_approximation=True)


def render_camera_selection(eval_dir,pred_dir,out_dir,*,clip_id=None,max_clips=0,ground_height_m=None,
                            max_depth_m=50.,fps=10.,crf=18,layout='both',horizon_steps=60):
    rows=read_jsonl(Path(eval_dir)/'selection.jsonl')
    if clip_id:rows=[r for r in rows if r['clip_id']==clip_id]
    if max_clips>0:rows=rows[:max_clips]
    if not rows:raise ValueError('No selected clips')
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    results=[]
    for row in rows:
        result=render_camera_clip(row,pred_dir,out/(row['clip_id']+'_camera.mp4'),
            ground_height_m=ground_height_m,max_depth_m=max_depth_m,fps=fps,crf=crf,
            layout=layout,horizon_steps=horizon_steps)
        results.append(result);print('[CAMERA RENDER]',json.dumps(result))
    (out/'diagnostics.json').write_text(json.dumps(results,indent=2))
    return results
