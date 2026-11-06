"""SE(3) geometry using the *same column-based rot6d convention* as cosmos3prep.

T_world_ego transforms ego-coordinate points into world coordinates.  Predicted
backward_framewise actions are translation in the previous ego coordinates and
rotation encoded as [R[:,0],R[:,1]]. No scale/rotation alignment is applied.
"""
import numpy as np


def check_pose_array(p, n=61):
    p = np.asarray(p, dtype=np.float64)
    if p.shape != (n, 4, 4) or not np.isfinite(p).all():
        raise ValueError(f"pose must be finite ({n},4,4); got {p.shape}")
    if not np.allclose(p[:, 3, :], (0., 0., 0., 1.), atol=1e-5):
        raise ValueError('Invalid homogeneous pose bottom row')
    R = p[:, :3, :3]
    if np.max(np.abs(R.transpose(0, 2, 1) @ R - np.eye(3))) > .003:
        raise ValueError('Pose contains non-orthonormal rotations')
    if not np.allclose(np.linalg.det(R), 1., atol=.003):
        raise ValueError('Pose contains reflections')
    return p


def check_actions(a, *, steps=60):
    a = np.asarray(a, dtype=np.float64)
    if a.shape != (steps, 9) or not np.isfinite(a).all():
        raise ValueError(f'action must be finite ({steps},9); got {a.shape}')
    return a


def rot6_decode(v):
    v = np.asarray(v, dtype=np.float64)
    a, b = v[..., :3], v[..., 3:6]
    an = np.linalg.norm(a, axis=-1, keepdims=True)
    if np.any(an < 1e-6):
        raise ValueError('Degenerate rot6d first vector')
    a = a / an
    b = b - np.sum(a*b, axis=-1, keepdims=True)*a
    bn = np.linalg.norm(b, axis=-1, keepdims=True)
    if np.any(bn < 1e-6):
        raise ValueError('Degenerate rot6d second vector')
    b = b / bn
    c = np.cross(a,b)
    return np.stack((a,b,c),axis=-1)


def encode_framewise(poses):
    p = check_pose_array(poses)
    delta = np.linalg.inv(p[:-1]) @ p[1:]
    rot = delta[:,:3,:3]
    return np.concatenate((delta[:,:3,3],rot[:,:,0],rot[:,:,1]),axis=-1)


def decode_actions(actions, convention='backward_framewise'):
    a = check_actions(actions)
    if convention not in ('backward_framewise','backward_anchored'):
        raise ValueError(f'Unsupported pose convention: {convention}')
    rotations = rot6_decode(a[:, 3:])
    out = np.tile(np.eye(4, dtype=np.float64), (len(a)+1,1,1))
    for i, (pos,R) in enumerate(zip(a[:,:3], rotations)):
        T = np.eye(4)
        T[:3,:3] = R
        T[:3,3] = pos
        out[i+1] = (out[i] if convention == 'backward_framewise' else out[0]) @ T
    return out


def localize(p):
    p = check_pose_array(p)
    return np.linalg.inv(p[0])[None] @ p


def rotation_geodesic_deg(ref_R, pred_R):
    err = np.swapaxes(ref_R, -1, -2) @ pred_R
    cosine = np.clip((np.trace(err,axis1=-2,axis2=-1) - 1.)/2., -1., 1.)
    # Geodesic stable near zero, using skew-vector and atan2.
    skew = np.stack((err[...,2,1]-err[...,1,2],
                     err[...,0,2]-err[...,2,0],
                     err[...,1,0]-err[...,0,1]),axis=-1)
    s = .5*np.linalg.norm(skew,axis=-1)
    return np.rad2deg(np.arctan2(s, cosine))


def metrics(gt_pose, pred_pose):
    """RRE/RTE are consecutive relative SE(3) errors; ATE is local-origin RMSE.

    No SE(3) or Sim(3) alignment beyond fixing the first frame to identity.
    """
    gt = localize(gt_pose)
    pr = localize(pred_pose)
    gt_rel = np.linalg.inv(gt[:-1]) @ gt[1:]
    pr_rel = np.linalg.inv(pr[:-1]) @ pr[1:]
    rre = rotation_geodesic_deg(gt_rel[:,:3,:3], pr_rel[:,:3,:3])
    rte = np.linalg.norm(gt_rel[:,:3,3]-pr_rel[:,:3,3],axis=-1)
    ate_frame = np.linalg.norm(gt[:,:3,3]-pr[:,:3,3],axis=-1)
    gt_r = gt[:,:3,:3]
    pr_r = pr[:,:3,:3]
    return dict(rre_deg=rre, rte_m=rte, ate_m=ate_frame,
                abs_rot_deg=rotation_geodesic_deg(gt_r,pr_r))
