"""Render GT/Pred poses with inset BEV in an H264 video (legacy ego mode)."""
import shutil
import subprocess
import tempfile
from pathlib import Path
import cv2
import numpy as np
from .data import read_jsonl,verify_row
from .geometry import decode_actions,metrics,localize
from .predictions import load_pred

GT_COLOR=(65,210,85)
PRED_COLOR=(75,80,240)


def _bev_points(gt,pred,size,pad=25):
    def xy(p):return np.stack((-p[:,1,3],-p[:,0,3]),axis=-1)
    aa,bb=xy(gt),xy(pred)
    points=np.concatenate((aa,bb));lo=points.min(axis=0);hi=points.max(axis=0)
    sc=min((size-2*pad)/v for v in np.maximum(hi-lo,1.))
    pix=np.rint((points-(lo+hi)/2)*sc+size*.5).astype(np.int32)
    return pix[:len(aa)],pix[len(aa):]


def _draw_bev(gt,pred,i,size):
    canvas=np.full((size,size,3),247,np.uint8)
    cv2.putText(canvas,'GT green | Pred red',(8,18),cv2.FONT_HERSHEY_SIMPLEX,.41,(35,35,35),1,cv2.LINE_AA)
    for pts,color in ((gt,GT_COLOR),(pred,PRED_COLOR)):
        cv2.polylines(canvas,[pts[:i+1].reshape(-1,1,2)],False,color,2,cv2.LINE_AA)
        cv2.circle(canvas,tuple(pts[i]),4,color,-1,cv2.LINE_AA)
    return canvas


def _encode_h264(frames,destination,fps,width,height,crf):
    if not shutil.which('ffmpeg'):raise RuntimeError('FFmpeg (libx264) is required')
    if width%2 or height%2:raise ValueError('H264 YUV420 requires even dimensions')
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-y',
         '-f','rawvideo','-pix_fmt','bgr24','-s',f'{width}x{height}',
         '-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf',str(crf),
         '-pix_fmt','yuv420p','-movflags','+faststart',str(destination)]
    with tempfile.TemporaryFile() as err:
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=err)
        try:
            for frame in frames:
                if frame.shape!=(height,width,3) or frame.dtype!=np.uint8:
                    raise ValueError(f'Unexpected render frame {frame.shape}/{frame.dtype}')
                proc.stdin.write(frame.tobytes())
            proc.stdin.close();code=proc.wait()
        except Exception:
            if proc.poll() is None:proc.kill()
            proc.wait();raise
        if code:
            err.seek(0);raise RuntimeError(err.read().decode(errors='replace'))
    if not destination.is_file() or not destination.stat().st_size:
        raise RuntimeError(f'Video was not written: {destination}')


def render_clip(row,pred_dir,out_file,*,fps=10.,crf=18):
    info=verify_row(row);gt=localize(info['gt_pose'])
    actions,_=load_pred(pred_dir,row['clip_id'])
    pred=decode_actions(actions,info['metadata']['pose_convention'])
    m=metrics(gt,pred)
    cap=cv2.VideoCapture(str(row['video']))
    if not cap.isOpened():raise OSError(row['video'])
    w,h=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size=min(300,int(h*.55),int(w*.40))
    if size<120:cap.release();raise ValueError('Video too small to render inset')
    gp,pp=_bev_points(gt,pred,size)
    def frames():
        for i in range(61):
            ok,bgr=cap.read()
            if not ok:raise ValueError('Video has fewer than 61 frames')
            inset=_draw_bev(gp,pp,i,size)
            bgr[12:12+size,w-12-size:w-12]=inset
            for j,t in enumerate((f'{row["clip_id"]} frame {i+1}/61',
                                  f'Position error {m["ate_m"][i]:.3f} m',
                                  f'Rotation error {m["abs_rot_deg"][i]:.3f} deg')):
                cv2.putText(bgr,t,(15,30+j*24),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1,cv2.LINE_AA)
            yield bgr
    try:_encode_h264(frames(),out_file,fps,w,h,crf)
    finally:cap.release()


def render_selection(eval_dir,pred_dir,out_dir,*,clip_id=None,max_clips=0,fps=10.,crf=18):
    rows=read_jsonl(Path(eval_dir)/'selection.jsonl')
    if clip_id:rows=[r for r in rows if r['clip_id']==clip_id]
    if max_clips:rows=rows[:max_clips]
    if not rows:raise ValueError('No clips selected')
    for row in rows:
        dst=Path(out_dir)/(row['clip_id']+'_gt_pred.mp4')
        render_clip(row,pred_dir,dst,fps=fps,crf=crf)
        print('[RENDER OK]',dst)
