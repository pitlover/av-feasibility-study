"""Temporal alignment onto an actual 10 Hz target grid.

Images at target times can be *estimated*, but not physically measured, when
there is no source exposure at that time. Never claim synthetic RGB as observed.
Absolute poses are interpolated in SE(3) (linear translation + SO(3) SLERP).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from cosmos3prep.core import Frame, frame_runs_kitti_native


@dataclass(frozen=True)
class SourcePair:
    left: Frame
    right: Frame
    alpha: float


def interpolate_se3(a, b, alpha):
    """Interpolate sensor-to-world transforms, T_world_sensor."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != (4,4) or b.shape != (4,4):
        raise ValueError('SE(3) poses must be 4x4')
    alpha = float(alpha)
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must lie within [0,1]; extrapolation forbidden')
    if alpha == 0: return a.copy()
    if alpha == 1: return b.copy()
    r = Slerp([0,1], Rotation.from_matrix(np.stack([a[:3,:3],b[:3,:3]])))([alpha]).as_matrix()[0]
    out = np.eye(4)
    out[:3,:3] = r
    out[:3,3] = (1-alpha)*a[:3,3] + alpha*b[:3,3]
    return out


def split_contiguous(frames, dataset, max_gap_s=0.16):
    """Split true acquisition sequences on gaps; do not pre-snap to any grid."""
    frames = sorted(frames, key=lambda f:f.t_s)
    if not frames: return []
    if dataset == 'kitti':
        # KITTI uses consecutive image indices as well as timestamps.
        runs = frame_runs_kitti_native(frames, max_dt_s=max_gap_s)
        return [r for r in runs if len(r) >= 2]
    result=[];current=[frames[0]]
    for f in frames[1:]:
        dt=float(f.t_s-current[-1].t_s)
        if 0 < dt <= max_gap_s:
            current.append(f)
        else:
            if len(current)>1: result.append(current)
            current=[f]
    if len(current)>1:result.append(current)
    return result


def make_exact_sample(frames, target_s, *, instant_tolerance_s=1e-6):
    """Return a sample with an image source pair and exact-time interpolated GT."""
    times = np.asarray([f.t_s for f in frames], dtype=np.float64)
    target_s=float(target_s)
    if target_s < times[0]-instant_tolerance_s or target_s > times[-1]+instant_tolerance_s:
        raise ValueError('Target requires extrapolation')
    j=int(np.searchsorted(times,target_s,side='left'))
    if j<len(frames) and abs(times[j]-target_s) <= instant_tolerance_s:
        left=right=frames[j];alpha=0.
    elif j>0 and abs(times[j-1]-target_s) <= instant_tolerance_s:
        left=right=frames[j-1];alpha=0.
    else:
        if j<=0 or j>=len(frames):raise ValueError('Missing interpolation bracket')
        left,right=frames[j-1],frames[j]
        dt=right.t_s-left.t_s
        if dt <= 0:raise ValueError('Source timestamps not increasing')
        alpha=float((target_s-left.t_s)/dt)
    if left is right:
        world=left.world_ego.copy()
        rig=left.ego_camera.copy()
        K=np.asarray(left.K).copy()
        distortion=np.asarray(left.distortion).copy()
    else:
        world=interpolate_se3(left.world_ego,right.world_ego,alpha)
        rig=interpolate_se3(left.ego_camera,right.ego_camera,alpha)
        K=(1-alpha)*np.asarray(left.K)+alpha*np.asarray(right.K)
        distortion=(1-alpha)*np.asarray(left.distortion)+alpha*np.asarray(right.distortion)
    return Frame(t_s=target_s, image=SourcePair(left,right,alpha),
                 world_ego=world, ego_camera=rig,
                 K=K, distortion=distortion, source_id=f'exact10hz:{target_s:.9f}',
                 needs_undistort=False)


def exact_grid_clips(frames, dataset, stride=60, max_gap_s=0.16,
                     fps=10, count=61):
    """Yield (start_grid_index, 61 interpolated Frames) using exact 100ms targets.

    A target frame is sometimes synthetic. No original exposure is overwritten.
    Rejects timing gaps and avoids temporal extrapolation.
    """
    if stride <= 0:raise ValueError('stride must be positive')
    dt=1. / float(fps)
    base_t=float(min(f.t_s for f in frames)) if frames else 0.
    for run in split_contiguous(frames,dataset,max_gap_s=max_gap_s):
        t0=float(run[0].t_s)
        global_tick=int(round((t0-base_t)/dt))
        ticks=int(np.floor((float(run[-1].t_s)-t0)/dt+1e-8))+1
        for tick in range(0,ticks-count+1,stride):
            target=t0+(tick+np.arange(count,dtype=np.float64))*dt
            if target[-1] > run[-1].t_s+1e-6:continue
            output=[make_exact_sample(run,t) for t in target]
            yield global_tick+tick,output


def blend_virtual(im0,im1,alpha,method='flow'):
    """Approximate video at an intermediate timestamp. Not observed GT imagery."""
    if im0.shape != im1.shape:raise ValueError('Source frame shapes differ')
    if alpha <= 0:return im0.copy()
    if alpha >= 1:return im1.copy()
    import cv2
    if method=='blend':
        return cv2.addWeighted(im0,1-alpha,im1,alpha,0)
    if method!='flow':raise ValueError('rgb interpolation: flow|blend')
    import cv2
    prev=cv2.cvtColor(im0,cv2.COLOR_BGR2GRAY)
    nxt=cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    flow01=cv2.calcOpticalFlowFarneback(prev,nxt,None,.5,3,17,3,5,1.2,0)
    flow10=cv2.calcOpticalFlowFarneback(nxt,prev,None,.5,3,17,3,5,1.2,0)
    h,w=prev.shape
    x,y=np.meshgrid(np.arange(w,dtype=np.float32),np.arange(h,dtype=np.float32))
    p0=cv2.remap(im0,x-alpha*flow01[...,0],y-alpha*flow01[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    p1=cv2.remap(im1,x-(1-alpha)*flow10[...,0],y-(1-alpha)*flow10[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    return cv2.addWeighted(p0,1-alpha,p1,alpha,0)
