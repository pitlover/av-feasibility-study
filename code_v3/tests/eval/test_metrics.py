import json
from pathlib import Path
import numpy as np
import pytest

from cosmos3eval.geometry import decode_actions,encode_framewise,metrics
from cosmos3eval.data import prepare,verify_row
from cosmos3eval.predictions import load_pred
from cosmos3eval.report import score


def mock_processed(tmp_path, name='kitti_09_000000'):
    root=tmp_path/'processed'
    c=root/'kitti'/'val'/name;c.mkdir(parents=True)
    p=np.tile(np.eye(4),(61,1,1));p[:,0,3]=np.arange(61)*.7
    # include planar small yaw; ensure rotation checks use actual matrices
    from math import sin,cos
    for i in range(61):
        ang=i*.005
        p[i,:3,:3]=[[cos(ang),-sin(ang),0],[sin(ang),cos(ang),0],[0,0,1]]
    a=encode_framewise(p)
    np.save(c/'poses_local_ego.npy',p)
    np.save(c/'actions_raw_9d.npy',a.astype('float32'))
    np.save(c/'timestamps_s.npy',np.arange(61)*.1)
    (c/'video.mp4').write_bytes(b'not a real mp4; prepare checks file presence only')
    metadata=dict(clip_id=name,dataset='kitti',split='val',sequence='09',fps=10,
        frame_count=61,action_count=60,pose_convention='backward_framewise',
        action_encoding='translation3 + rot6d_column_based',action_normalization='none',
        pose_frame='velodyne')
    (c/'metadata.json').write_text(json.dumps(metadata))
    idx=dict(clip_id=name,dataset='kitti',split='val',sequence='09',video=str(c/'video.mp4'),
        meta=str(c/'metadata.json'),action_file=str(c/'actions_raw_9d.npy'))
    (root/'index.jsonl').write_text(json.dumps(idx)+'\n')
    return root,idx,p,a


def test_exact_oracle():
    p=np.tile(np.eye(4),(61,1,1));p[:,0,3]=np.arange(61)*.5
    a=encode_framewise(p)
    pred=decode_actions(a)
    m=metrics(p,pred)
    assert max(m['rre_deg'])<1e-7
    assert max(m['rte_m'])<1e-8
    assert max(m['ate_m'])<1e-8


def test_translation_shift_rte_ate():
    p=np.tile(np.eye(4),(61,1,1));p[:,0,3]=np.arange(61)*.1
    a=encode_framewise(p); b=a.copy();b[:,0]+=.01
    pred=decode_actions(b)
    m=metrics(p,pred)
    assert np.allclose(m['rte_m'],.01,atol=1e-8)
    assert np.isclose(m['ate_m'][-1],.6)


def test_rotation_rre():
    p=np.tile(np.eye(4),(61,1,1));a=encode_framewise(p)
    b=a.copy(); ang=np.deg2rad(1.)
    b[:,3:9]=np.array([np.cos(ang),np.sin(ang),0.,-np.sin(ang),np.cos(ang),0.])
    m=metrics(p,decode_actions(b))
    assert np.allclose(m['rre_deg'],1.,atol=1e-8)


def test_prepare_and_score(tmp_path):
    root,row,p,a=mock_processed(tmp_path)
    v=verify_row(row); assert v['max_gt_roundtrip_m']<1e-4
    ev=tmp_path/'eval';prepare(root,ev,['kitti'])
    req=json.loads((ev/'inference_requests.jsonl').read_text())
    assert req['model_mode']=='inverse_dynamics' and req['action_chunk_size']==60
    pred=tmp_path/'pred';(pred/row['clip_id']).mkdir(parents=True)
    (pred/row['clip_id']/'sample_outputs.json').write_text(json.dumps({'status':'success','action':a.tolist()}))
    res=score(ev,pred,model='test',plots=False)
    assert res['overall']['ate_m_rmse']<1e-5
    assert res['overall']['rre_deg_mean']<1e-4


def test_bad_prediction_not_masked(tmp_path):
    root,row,p,a=mock_processed(tmp_path)
    ev=tmp_path/'eval';prepare(root,ev,['kitti'])
    with pytest.raises(FileNotFoundError):score(ev,tmp_path/'missing',plots=False)


def test_dcp_infer_cmd_not_silently_accepted(tmp_path):
    from cosmos3eval.cli import infer_cmd
    from argparse import Namespace
    fw=tmp_path/'framework';(fw/'cosmos_framework').mkdir(parents=True)
    ev=tmp_path/'eval';ev.mkdir();(ev/'inference_requests.jsonl').write_text('{}\n')
    ck=tmp_path/'dcp';ck.mkdir();(ck/'.metadata').touch()
    with pytest.raises(ValueError, match='Raw DCP'):
        infer_cmd(Namespace(framework_root=str(fw),eval_dir=str(ev),pred_dir=str(tmp_path/'pred'),checkpoint=str(ck),gpus=1,seed=0,execute=False))
