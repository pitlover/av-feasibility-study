import numpy as np
from cosmos3virtual.normalization import FORMAT,transform_actions


def test_custom_normalizer_translation_only_roundtrip():
    norm={'format':FORMAT,'frame':'camera','method':'translation3_only',
          'mean_9d':[0,0,1,0,0,0,0,0,0],
          'scale_9d':[1,1,2,1,1,1,1,1,1]}
    a=np.zeros((60,9),np.float32);a[:,2]=1.4;a[:,3]=1.;a[:,7]=1.
    n=transform_actions(a,norm)
    assert np.array_equal(n[:,3:],a[:,3:])
    np.testing.assert_allclose(transform_actions(n,norm,inverse=True),a)


def test_v2_preprocess_cli_forwarding(monkeypatch):
    from cosmos3virtual import cli
    import cosmos3prep.cli as prep
    args=[]
    monkeypatch.setattr(prep,'main',lambda x:args.extend(x))
    cli.main(['av2','--root','/foo','--out','/bar','--time-mode','native',
              '--camera','stereo_front_left','--width','640','--height','384','--hfov','45'])
    assert args[0]=='av2'
    assert args[args.index('--action-frame')+1]=='camera'
    assert args[args.index('--image-mode')+1]=='virtual'


def test_v2_camera_render_cli(monkeypatch,tmp_path):
    from cosmos3eval import cli,camera_render
    seen={}
    monkeypatch.setattr(camera_render,'render_camera_selection',lambda *args,**kw:seen.update(kw))
    cli.main(['render','--eval-dir',str(tmp_path/'e'),'--pred-dir',str(tmp_path/'p'),
              '--mode','camera','--model','stage2_av2_raw',
              '--layout','both','--horizon-steps','30','--ego-height-m','1.7'])
    assert seen['layout']=='both'
    assert seen['horizon_steps']==30
    assert seen['ground_height_m']==1.7
