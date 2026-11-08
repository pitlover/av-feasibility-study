"""SE(3) trajectory geometry for camera/ego relative Cosmos actions."""
import numpy as np


def check_pose_array(p, n=61):
    p=np.asarray(p,np.float64)
    if p.shape!=(n,4,4) or not np.isfinite(p).all():
        raise ValueError(f'pose must be finite ({n},4,4); got {p.shape}')
    if not np.allclose(p[:,3,:],[0.,0.,0.,1.],atol=1e-5):
        raise ValueError('Invalid homogeneous bottom row')
    R=p[:,:3,:3]
    if np.max(np.abs(R.transpose(0,2,1) @ R-np.eye(3)))>.003 or not np.allclose(np.linalg.det(R),1.,atol=.003):
        raise ValueError('Invalid rotation matrix')
    return p


def check_actions(a, steps=60):
    a=np.asarray(a,np.float64)
    if a.shape!=(steps,9) or not np.isfinite(a).all():
        raise ValueError(f'action must be finite ({steps},9); got {a.shape}')
    return a


def rot6_decode(v):
    v=np.asarray(v,np.float64)
    a,b=v[...,:3],v[...,3:6]
    an=np.linalg.norm(a,axis=-1,keepdims=True)
    if np.any(an<1e-6):raise ValueError('Degenerate 6D rotation')
    a=a/an
    b=b-(a*b).sum(axis=-1,keepdims=True)*a
    bn=np.linalg.norm(b,axis=-1,keepdims=True)
    if np.any(bn<1e-6):raise ValueError('Degenerate 6D rotation')
    b=b/bn
    return np.stack((a,b,np.cross(a,b)),axis=-1)


def encode_framewise(poses):
    p=check_pose_array(poses)
    d=np.linalg.inv(p[:-1])@p[1:]
    return np.concatenate((d[:,:3,3],d[:,:3,0],d[:,:3,1]),axis=-1)


def decode_actions(actions, convention='backward_framewise'):
    a=check_actions(actions)
    if convention not in ('backward_framewise','backward_anchored'):
        raise ValueError(convention)
    R=rot6_decode(a[:,3:])
    out=np.tile(np.eye(4),(len(a)+1,1,1))
    for i,(p,r) in enumerate(zip(a[:,:3],R)):
        step=np.eye(4);step[:3,:3]=r;step[:3,3]=p
        out[i+1]=(out[i] if convention=='backward_framewise' else out[0])@step
    return out


def localize(p):
    p=check_pose_array(p)
    return np.linalg.inv(p[0])[None]@p


def rotation_geodesic_deg(ref_R,pred_R):
    err=np.swapaxes(ref_R,-1,-2)@pred_R
    cosine=np.clip((np.trace(err,axis1=-2,axis2=-1)-1.)/2.,-1.,1.)
    skew=np.stack((err[...,2,1]-err[...,1,2],err[...,0,2]-err[...,2,0],err[...,1,0]-err[...,0,1]),axis=-1)
    return np.rad2deg(np.arctan2(.5*np.linalg.norm(skew,axis=-1),cosine))


def metrics(gt_pose,pred_pose):
    gt=localize(gt_pose);pr=localize(pred_pose)
    gr=np.linalg.inv(gt[:-1])@gt[1:]
    rr=np.linalg.inv(pr[:-1])@pr[1:]
    return dict(rre_deg=rotation_geodesic_deg(gr[:,:3,:3],rr[:,:3,:3]),
                rte_m=np.linalg.norm(gr[:,:3,3]-rr[:,:3,3],axis=-1),
                ate_m=np.linalg.norm(gt[:,:3,3]-pr[:,:3,3],axis=-1),
                abs_rot_deg=rotation_geodesic_deg(gt[:,:3,:3],pr[:,:3,:3]))
