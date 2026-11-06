import json
import subprocess
from pathlib import Path
import numpy as np
import pytest
from cosmos3av.data import load_index, mix_rows, select_stage, preflight
from cosmos3av.cli import main
from cosmos3av.framework_dataset import AVInverseDataset


def fake_processed(tmp_path):
    root=tmp_path/'processed'
    root.mkdir()
    rows=[]
    for ds in ['kitti','waymo','av2']:
        for split in ['train','val']:
            for seq in range(2 if split=='train' else 1):
                ident=f'{ds}_{split}_{seq}'
                p=root/ds/split/ident
                p.mkdir(parents=True)
                np.save(p/'actions_raw_9d.npy',np.zeros((60,9),dtype=np.float32))
                np.save(p/'timestamps_s.npy',np.arange(61,dtype=np.float64)/10)
                meta=dict(clip_id=ident,dataset=ds,split=split,sequence=f'{ds}_{seq}',
                   pose_convention='backward_framewise',action_normalization='none',
                   fps=10,frame_count=61,action_count=60)
                (p/'metadata.json').write_text(json.dumps(meta))
                vid=p/'video.mp4'
                # Small 61f H.264 video (fast)
                subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y',
                    '-f','lavfi','-i','color=size=32x32:rate=10:color=black',
                    '-frames:v','61','-c:v','libx264','-pix_fmt','yuv420p',str(vid)],check=True)
                rows.append(dict(clip_id=ident,dataset=ds,split=split,sequence=f'{ds}_{seq}',
                   video=str(vid),meta=str(p/'metadata.json'),action_file=str(p/'actions_raw_9d.npy')))
    (root/'index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    ckpt=tmp_path/'checkpoint'; ckpt.mkdir()
    vae=tmp_path/'Wan2.2_VAE.pth'; vae.write_bytes(b'not a real checkpoint - testing plan only')
    return root,ckpt,vae


def test_dataset_selection_and_ratios(tmp_path):
    root,_,_=fake_processed(tmp_path)
    rows=load_index(root)
    assert len(select_stage(rows,'pretrain',['kitti'],'train'))==2
    assert len(select_stage(rows,'pretrain',['kitti','waymo'],'train'))==4
    assert len(select_stage(rows,'posttrain',['av2'],'train'))==2
    with pytest.raises(ValueError): select_stage(rows,'pretrain',['av2'],'train')
    with pytest.raises(ValueError): select_stage(rows,'posttrain',['waymo'],'train')
    chosen=select_stage(rows,'pretrain',['kitti','waymo'])
    a=mix_rows(chosen,{'kitti':1,'waymo':3},40,42)
    assert [r['dataset'] for r in a].count('waymo')==30
    assert a==mix_rows(chosen,{'kitti':1,'waymo':3},40,42)


def test_preflight_and_mp4(tmp_path):
    root,_,_=fake_processed(tmp_path)
    row=load_index(root)[0]
    assert preflight(row,check_video=True)['mean_dt_s']==pytest.approx(.1)
    av=AVInverseDataset(str(root/'index.jsonl'),validate_on_load=True)
    assert len(av.get_shuffle_blocks())==9
    assert len(av)==9


def test_generate_plan_for_stages(tmp_path):
    root,ckpt,vae=fake_processed(tmp_path)
    p=tmp_path/'run1'
    main(['plan','--stage','pretrain','--datasets','kitti','waymo',
         '--weights','kitti:1,waymo:3', '--processed-root',str(root),
         '--init-checkpoint',str(ckpt),'--wan-vae',str(vae),'--outdir',str(p),
         '--epoch-clips','40','--gpus','2','--max-iters','5','--check-video'])
    detail=json.loads((p/'run.json').read_text())
    assert detail['n_train']==40 and detail['n_val']==2
    train=[json.loads(x) for x in (p/'train.jsonl').read_text().splitlines()]
    assert sum(r['dataset']=='waymo' for r in train)==30
    assert 'action_inverse_dynamics_av_edge' in (p/'sft_config.toml').read_text()
    p2=tmp_path/'run2'
    main(['plan','--stage','posttrain','--datasets','av2',
         '--processed-root',str(root), '--init-checkpoint',str(ckpt),
         '--wan-vae',str(vae),'--outdir',str(p2),'--epoch-clips','6'])
    r=json.loads((p2/'run.json').read_text())
    assert r['datasets']==['av2']
    assert all(json.loads(x)['dataset']=='av2' for x in (p2/'train.jsonl').read_text().splitlines())


def test_time_grid_reject_bad_gap(tmp_path):
    root,_,_=fake_processed(tmp_path)
    r=load_index(root)[0]
    ts=np.load(Path(r['meta']).parent/'timestamps_s.npy')
    ts[30:]+=0.2
    np.save(Path(r['meta']).parent/'timestamps_s.npy',ts)
    with pytest.raises(ValueError, match='dt'):
        preflight(r)
