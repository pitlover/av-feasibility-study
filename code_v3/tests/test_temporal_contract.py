import json
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
import cv2
from cosmos3prep.core import Frame
from cosmos3virtual.temporal import exact_grid_clips, interpolate_se3, SourcePair
from cosmos3virtual.camera import VirtualSpec
from cosmos3virtual.build import make_clip
from cosmos3virtual.normalization import fit_train_maxabs, transform_action, inverse_action
from cosmos3virtual.cli import validate, denorm_preds


def _frames(tmp_path, n=65, dt=.104, ds='kitti'):
    spec=VirtualSpec(64,48,45)
    f=[]
    for i in range(n):
        p=tmp_path/f'{i:06d}.png'
        if ds=='kitti':
            im=np.full((48,64,3),i%255,dtype=np.uint8)
            assert cv2.imwrite(str(p),im)
        T=np.eye(4)
        T[0,3]=i*dt*3
        f.append(Frame(t_s=i*dt,image=p,world_ego=T,ego_camera=np.eye(4),
                       K=spec.K,distortion=np.zeros(5),source_id=str(p)))
    return f,spec


def test_slerp():
    A=np.eye(4);B=np.eye(4)
    a=np.pi/2
    B[:3,:3]=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
    B[:3,3]=[2,0,0]
    M=interpolate_se3(A,B,.5)
    assert np.allclose(M[:3,3],[1,0,0])
    assert np.allclose(M[:3,:3]@np.array([1,0,0]),[2**-.5,2**-.5,0],atol=1e-6)


def test_exact_kitti_time_and_poses(tmp_path):
    frames,spec=_frames(tmp_path)
    tick,clip=next(exact_grid_clips(frames,'kitti',stride=60))
    assert tick==0
    ts=np.array([f.t_s for f in clip])
    assert len(ts)==61
    assert np.max(np.abs(np.diff(ts)-.1))<1e-10
    assert abs(clip[-1].world_ego[0,3]-18)<1e-7
    assert sum(f.image.left is not f.image.right for f in clip)>0


def test_temporal_gap_rejection(tmp_path):
    frames,_=_frames(tmp_path,n=40)
    # Insert 1-second gap; no valid clip can cross it
    frames[20:]=[Frame(t_s=f.t_s+1,image=f.image,world_ego=f.world_ego,
                       ego_camera=f.ego_camera,K=f.K,distortion=f.distortion,
                       source_id=f.source_id) for f in frames[20:]]
    assert not list(exact_grid_clips(frames,'kitti'))


def test_end_to_end_virtual_gt(tmp_path):
    frames,spec=_frames(tmp_path)
    _,clip=next(exact_grid_clips(frames,'kitti'))
    out=tmp_path/'output'
    created=make_clip(clip,dataset='kitti',sequence='00',split='train',first_idx=0,
                      spec=spec,out_root=out,rgb_interpolation='blend',time_mode='exact',
                      min_valid_fraction=.95)
    assert created
    folder=out/'kitti/train/kitti_00_000000'
    m=json.loads((folder/'metadata.json').read_text())
    assert m['time_mode']=='exact'
    assert m['synthetic_rgb_frames']>0
    assert m['clip_timebase']=='exact_100ms_with_rgb_temporal_interpolation'
    assert np.allclose(np.diff(np.load(folder/'timestamps_s.npy')),.1)
    assert np.load(folder/'actions_raw_9d.npy').shape==(60,9)
    from cosmos3virtual.build import rebuild_virtual_index
    rebuild_virtual_index(out)
    validate(SimpleNamespace(out=str(out),max_clips=0))


def test_norm_train_only_and_inverse(tmp_path):
    arr=np.zeros((60,9));arr[:,:3]=[2,3,4];arr[:,3]=1;arr[:,7]=1
    train=tmp_path/'train.npy';np.save(train,arr)
    val=tmp_path/'val.npy';np.save(val,arr*100)
    rows=[dict(dataset='kitti',split='train',clip_id='train',action_file=str(train)),
          dict(dataset='av2',split='val',clip_id='val',action_file=str(val))]
    norm=fit_train_maxabs(rows,['kitti'])
    assert np.allclose(norm['xyz_scale_m'],[2,3,4])
    assert np.allclose(inverse_action(transform_action(arr,norm),norm),arr)


def test_denorm_requires_training_contract(tmp_path):
    p=tmp_path/'pred';(p/'c').mkdir(parents=True)
    arr=np.zeros((60,9));arr[:,2]=.5
    (p/'c/sample_outputs.json').write_text(json.dumps({'outputs':[{'content':{'action':arr.tolist()}}]}))
    norm=fit_train_maxabs([dict(dataset='kitti',split='train',clip_id='c',action_file=str(tmp_path/'act.npy'))],['kitti']) if False else None
    rn=tmp_path/'train';rn.mkdir()
    import hashlib
    norm_data={'schema':'cosmos3_av_translation_maxabs_v1','xyz_scale_m':[1,2,3]}
    normalizer=rn/'action_normalizer.json';normalizer.write_text(json.dumps(norm_data))
    info={'action_normalizer':str(normalizer),'action_normalizer_sha256':hashlib.sha256(normalizer.read_bytes()).hexdigest()}
    (rn/'run.json').write_text(json.dumps(info))
    args=SimpleNamespace(pred_dir=str(p),out=str(tmp_path/'raw'),stage_run=str(rn),assume_normalized_output=False)
    with pytest.raises(ValueError,match='assume-normalized'):
        denorm_preds(args)
    args.assume_normalized_output=True
    denorm_preds(args)
    res=json.loads((tmp_path/'raw/c/sample_outputs.json').read_text())
    assert np.allclose(np.asarray(res['outputs'][0]['content']['action'])[:,2],1.5)
    with pytest.raises(ValueError):
        denorm_preds(SimpleNamespace(pred_dir=str(tmp_path/'raw'),out=str(tmp_path/'new'),stage_run=str(rn),assume_normalized_output=True))


def test_stage2_inherits_frozen_stage1_normalizer_with_native_timestamps(tmp_path):
    from cosmos3virtual.build import rebuild_virtual_index
    from cosmos3av.cli import plan
    frames,spec=_frames(tmp_path)
    clip=frames[:61]  # actual source exposures at 104ms, no artificial RGB interpolation
    src=tmp_path/'processed_v2'
    for dataset,split,seq in [('kitti','train','00'),('av2','train','av2log')]:
        make_clip(clip,dataset=dataset,sequence=seq,split=split,first_idx=0,
                  spec=spec,out_root=src,time_mode='native',
                  min_valid_fraction=.95)
    rebuild_virtual_index(src)
    rows=[json.loads(line) for line in (src/'index.jsonl').read_text().splitlines()]
    norm=fit_train_maxabs(rows,['kitti'])
    norm_path=tmp_path/'stage1_norm.json';norm_path.write_text(json.dumps(norm))
    base_ckpt=tmp_path/'edge_dcp';base_ckpt.mkdir()
    vae=tmp_path/'WanVAE.pth';vae.write_bytes(b'x')
    kw=dict(processed_root=str(src),outdir=str(tmp_path/'stage1'),
            init_checkpoint=str(base_ckpt),wan_vae=str(vae),
            action_normalizer=str(norm_path),ack_custom_action_normalizer=True,
            max_dt_error_s=.07,full_preflight=True,check_video=False,epoch_clips=1,
            max_iters=10,save_every=10,gpus=4,lr=1e-5,seed=42,
            overwrite_plan=False,stage1_run=None,allow_external_stage1_checkpoint=False)
    plan(SimpleNamespace(stage='pretrain',datasets=['kitti'],weights='',**kw))
    stage1=tmp_path/'stage1'
    stage1_ckpt=stage1/'outputs/checkpoints/iter_10';stage1_ckpt.mkdir(parents=True)
    stage2_kw={**kw,'outdir':str(tmp_path/'stage2'),
               'init_checkpoint':str(stage1_ckpt),
               'action_normalizer':'',
               'stage1_run':str(stage1)}
    plan(SimpleNamespace(stage='posttrain',datasets=['av2'],weights='',**stage2_kw))
    j1=json.loads((stage1/'run.json').read_text())
    j2=json.loads((tmp_path/'stage2/run.json').read_text())
    assert j1['action_normalizer_sha256']==j2['action_normalizer_sha256']
    assert json.loads((tmp_path/'stage2/action_normalizer.json').read_text())['source_split']=='train_only'
    # Cannot change Stage2 stats even with a different file that has valid schema.
    alternate=tmp_path/'other_norm.json'
    altered=norm.copy();altered['xyz_scale_m']=[20,20,20]
    alternate.write_text(json.dumps(altered))
    with pytest.raises(ValueError,match='cannot fit/switch'):
        plan(SimpleNamespace(stage='posttrain',datasets=['av2'],weights='',
                             **{**stage2_kw,'outdir':str(tmp_path/'stage2_wrong'),
                                'action_normalizer':str(alternate)}))


def test_native_kitti_preserves_actual_rgb_and_gt(tmp_path):
    frames,spec=_frames(tmp_path, n=61, dt=.104)
    from cosmos3virtual.build import rebuild_virtual_index
    out=tmp_path/'native'
    assert make_clip(frames,dataset='kitti',sequence='00',split='train',first_idx=0,
                     spec=spec,out_root=out,time_mode='native',min_valid_fraction=.95)
    folder=out/'kitti/train/kitti_00_000000'
    m=json.loads((folder/'metadata.json').read_text())
    assert m['time_mode']=='native'
    assert m['synthetic_rgb_frames']==0
    assert np.allclose(np.diff(np.load(folder/'timestamps_s.npy')),.104)
    assert m['actual_capture_duration_s'] > 6.0
    assert np.isclose(m['actual_capture_duration_s'],6.24)
    assert m['clip_timebase']=='nominal_10fps_native_timestamps'
    rebuild_virtual_index(out)
    validate(SimpleNamespace(out=str(out),max_clips=0,max_dt_error_s=.07))


def test_native_validation_rejects_bad_frame_interval(tmp_path):
    frames,spec=_frames(tmp_path, n=61, dt=.145)
    from cosmos3virtual.build import rebuild_virtual_index
    out=tmp_path/'native2'
    assert make_clip(frames,dataset='kitti',sequence='00',split='train',first_idx=0,
                     spec=spec,out_root=out,time_mode='native',min_valid_fraction=.95)
    rebuild_virtual_index(out)
    with pytest.raises(ValueError, match='exceeds tolerance'):
        validate(SimpleNamespace(out=str(out),max_clips=0,max_dt_error_s=.03))
