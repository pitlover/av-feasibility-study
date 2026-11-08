from __future__ import annotations
import json
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import cv2
import pytest

from cosmos3prep.core import Frame
from cosmos3prep.geometry import decode_local
from cosmos3virtual.camera import VirtualSpec, optical_from_source, R_WAYMO_CAMERA_FROM_OPTICAL,render_virtual
from cosmos3virtual.build import make_clip, rebuild_virtual_index
from cosmos3virtual.normalization import fit_train_maxabs, transform_action, inverse_action, read_norm


def test_virtual_K_and_waymo_optical():
    k=VirtualSpec(640,384,45).K
    assert k[0,0] == pytest.approx(772.5483399593904)
    assert k[1,1] == pytest.approx(k[0,0])
    assert k[0,2] == 320 and k[1,2]==192
    assert np.linalg.det(R_WAYMO_CAMERA_FROM_OPTICAL) == pytest.approx(1)
    # Waymo sensor +X forward -> OpenCV virtual +Z forward.
    v=optical_from_source('waymo')[:3,:3]@np.array([0.,0.,1.])
    assert np.allclose(v, [1.,0.,0.])


def test_remap_identity():
    h,w=384,640
    image=np.random.default_rng(1).integers(0,255,(h,w,3),dtype=np.uint8)
    k=VirtualSpec(w,h,45).K
    out,coverage=render_virtual(image,k,np.zeros(5),k,(w,h), min_valid_fraction=.99)
    assert coverage>=.99
    assert np.mean(np.abs(out.astype(float)-image.astype(float))) < .02


@pytest.mark.parametrize('ds', ['kitti','waymo','av2'])
def test_virtual_clip_and_train_only_norm(tmp_path,ds):
    spec=VirtualSpec(640,384,45)
    src=tmp_path/'source.png'
    cv2.imwrite(str(src),np.full((384,640,3),(45,90,130),np.uint8))
    t_src_to_ego=np.eye(4)
    if ds=='waymo':
        # Identity source extrinsic -> x-forward in ego sensor coords
        pass
    frames=[]
    for i in range(61):
        world=np.eye(4)
        world[0,3] = float(i)*.6  # ego vehicle forward
        frames.append(Frame(
            t_s=float(i)*.1, image=src, world_ego=world, ego_camera=t_src_to_ego,
            K=spec.K,distortion=np.zeros(5), source_id=f'{ds}/{i:06d}.png'))
    ok=make_clip(frames,dataset=ds,sequence='seq00',split='train',first_idx=0,
                 spec=spec,out_root=tmp_path/'processed',encoder='local')
    assert ok
    n=rebuild_virtual_index(tmp_path/'processed')
    assert n==1
    clip=tmp_path/'processed'/ds/'train'/f'{ds}_seq00_000000'
    poses=np.load(clip/'poses_local_virtual_camera.npy')
    actions=np.load(clip/'actions_raw_9d.npy')
    assert actions.shape==(60,9)
    assert poses.shape==(61,4,4)
    # For Waymo, virtual optical forward +Z corresponds to ego +X.
    if ds=='waymo':
        assert actions[:,2].mean()==pytest.approx(.6,abs=1e-4)
        assert np.abs(actions[:,0]).mean()<1e-5
    meta=json.loads((clip/'metadata.json').read_text())
    assert meta['pose_frame']=='virtual_camera'
    assert meta['action_normalization']=='none'
    rows=[json.loads(s) for s in (tmp_path/'processed'/'index.jsonl').read_text().splitlines()]
    norm=fit_train_maxabs(rows,[ds])
    p=tmp_path/'norm.json';p.write_text(json.dumps(norm))
    assert read_norm(p)['xyz_scale_m'][2] > 0
    normalized=transform_action(actions,norm)
    restored=inverse_action(normalized,norm)
    assert np.allclose(actions,restored,atol=1e-6)
    assert np.max(np.abs(normalized[:,:3]))<=1+1e-5
    assert np.allclose(actions[:,3:],normalized[:,3:])
    gt=decode_local(actions,'backward_framewise')
    assert np.max(np.abs(gt[:,:3,3]-poses[:,:3,3]))<.002


def test_train_only_no_val_leakage(tmp_path):
    train=np.zeros((60,9),np.float32);train[:,0]=.2
    val=np.zeros((60,9),np.float32);val[:,0]=100
    a=tmp_path/'a.npy';b=tmp_path/'b.npy'
    np.save(a,train);np.save(b,val)
    rows=[dict(dataset='kitti',split='train',clip_id='a',action_file=str(a)),dict(dataset='kitti',split='val',clip_id='b',action_file=str(b))]
    norm=fit_train_maxabs(rows,['kitti'])
    assert norm['xyz_scale_m'][0]==pytest.approx(.2,abs=1e-5)


def test_camera_transform_change_world_ego(tmp_path):
    # Input Waymo ego pose X-forward -> camera local Z-forward; prove not just label renaming.
    T=optical_from_source('waymo')
    world=np.eye(4);world[0,3]=1
    change=np.linalg.inv(T) @ world @ T
    assert np.allclose(change[:3,3],[0,0,1.])


def test_training_plan_and_evaluation_virtual(tmp_path):
    from cosmos3virtual.build import make_clip,rebuild_virtual_index
    from cosmos3eval.data import verify_row
    from cosmos3av.cli import main as train_main
    spec=VirtualSpec(640,384,45)
    img=tmp_path/'img.png';cv2.imwrite(str(img),np.zeros((384,640,3),np.uint8))
    rows=[]
    for ds in ('kitti','waymo'):
        frames=[]
        for i in range(61):
            T=np.eye(4);T[0,3]=.2*i
            frames.append(Frame(t_s=i*.1,image=img,world_ego=T,
                                ego_camera=np.eye(4),K=spec.K,source_id=f'{ds}/{i:06d}.png'))
        assert make_clip(frames,dataset=ds,sequence='log',split='train',first_idx=0,
                         spec=spec,out_root=tmp_path/'processed')
    rebuild_virtual_index(tmp_path/'processed')
    idx=[json.loads(l) for l in (tmp_path/'processed'/'index.jsonl').read_text().splitlines()]
    v=verify_row(idx[0]);assert v['metadata']['pose_frame']=='virtual_camera'
    norm=fit_train_maxabs(idx,['kitti','waymo'])
    normpath=tmp_path/'norm.json';normpath.write_text(json.dumps(norm))
    ckpt=tmp_path/'checkpoint';ckpt.mkdir()
    vae=tmp_path/'vae.pth';vae.touch()
    run=tmp_path/'run'
    train_main(['plan','--stage','pretrain','--datasets','kitti','waymo',
                '--processed-root',str(tmp_path/'processed'),
                '--init-checkpoint',str(ckpt),'--wan-vae',str(vae),
                '--outdir',str(run),'--gpus','1','--max-iters','5',
                '--epoch-clips','2','--action-normalizer',str(normpath),
                '--ack-custom-action-normalizer','--full-preflight'])
    d=json.loads((run/'run.json').read_text())
    assert d['action_normalizer']==str(run/'action_normalizer.json')
    assert (run/'action_normalizer.json').is_file()
    assert len((run/'train.jsonl').read_text().splitlines())==2
    assert 'action_inverse_dynamics_av_edge' in (run/'sft_config.toml').read_text()


def test_denorm_output_json(tmp_path):
    from cosmos3virtual.cli import main
    normpath=tmp_path/'norm.json'
    normpath.write_text(json.dumps(dict(schema='cosmos3_av_translation_maxabs_v1',xyz_scale_m=[2.,3.,4.])))
    s=tmp_path/'pred'/'clip0';s.mkdir(parents=True)
    a=np.zeros((60,9),dtype=np.float64);a[:,2]=.25;a[:,3]=1.;a[:,7]=1.
    (s/'sample_outputs.json').write_text(json.dumps(dict(outputs=[dict(content=dict(action=a.tolist()))])))
    import hashlib
    run=tmp_path/'stage1';run.mkdir()
    (run/'run.json').write_text(json.dumps({'action_normalizer':str(normpath),
        'action_normalizer_sha256':hashlib.sha256(normpath.read_bytes()).hexdigest()}))
    main(['denormalize-pred','--pred-dir',str(tmp_path/'pred'), '--stage-run',str(run),
          '--assume-normalized-output','--out',str(tmp_path/'raw')])
    b=json.loads((tmp_path/'raw'/'clip0'/'sample_outputs.json').read_text())
    assert b['outputs'][0]['content']['action'][0][2]==pytest.approx(1.)
