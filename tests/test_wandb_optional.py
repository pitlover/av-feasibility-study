"""CPU-only optional W&B integration regression tests (no network required)."""
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from cosmos3av.cli import _toml
from cosmos3eval.wandb_logging import _scalars, log_evaluation


def _summary():
    return {'overall': {'model': 's1_full_raw_v1_iter_3000', 'oracle': False,
            'rre_deg_mean': .258, 'rte_m_mean': .035, 'ate_m_rmse': 2.174,
            'final_position_error_m_mean': 3.553, 'n_clips': 40, 'n_failed': 0,
            'metrics_definition': 'relative SE3'},
            'by_dataset': {'kitti': {'n_clips': 20, 'rre_deg_mean': .263,
                        'rte_m_mean': .043, 'ate_m_rmse': 2.39},
                           'waymo': {'n_clips': 20, 'rre_deg_mean': .254,
                        'rte_m_mean': .028, 'ate_m_rmse': 1.93}}}


def test_toml_modes():
    for mode in ('online','offline','disabled'):
        text = _toml('pretrain','foo',3000,1e-5,500,4,mode)
        assert f'wandb_mode = "{mode}"' in text
        assert 'group = "pretrain"' in text
    assert 'wandb_mode = "disabled"' in _toml('posttrain','bar',10,1e-5,5,4)
    with pytest.raises(ValueError):
        _toml('pretrain','bar',10,1e-5,5,4,'enabled')


def test_disabled_does_not_import_wandb(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, 'wandb', raising=False)
    assert log_evaluation(summary=_summary(), metrics_dir=tmp_path,
                          mode='disabled') is None
    assert not (tmp_path/'wandb').exists()


def test_oracle_cannot_be_logged(tmp_path):
    data = _summary()
    data['overall']['oracle'] = True
    with pytest.raises(ValueError):
        log_evaluation(summary=data, metrics_dir=tmp_path, mode='offline')


class FakeRun:
    def __init__(self):
        self.logs=[]
        self.summary={}
        self.artifacts=[]
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def log(self, metrics): self.logs.append(metrics)
    def log_artifact(self, artifact): self.artifacts.append(artifact)


class FakeArtifact:
    def __init__(self, name, type):
        self.name=name
        self.type=type
        self.files=[]
    def add_file(self, p): self.files.append(p)


def test_online_offline_evaluation_logging(tmp_path, monkeypatch):
    calls=[]
    runs=[]
    def fake_init(**kw):
        calls.append(kw)
        run=FakeRun()
        runs.append(run)
        return run
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(
        init=fake_init, Artifact=FakeArtifact,
        Table=lambda **kw: ('fake_table',kw)))
    with (tmp_path/'per_clip.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=['clip_id','dataset','rre_deg_mean','rte_m_mean','ate_m_rmse'])
        writer.writeheader()
        writer.writerow({'clip_id':'kitti_09_000000', 'dataset':'kitti',
                         'rre_deg_mean':'.2','rte_m_mean':'.03','ate_m_rmse':'1.3'})
    (tmp_path/'summary.json').write_text(json.dumps(_summary()))
    for mode in ('online','offline'):
        logged=log_evaluation(summary=_summary(), metrics_dir=tmp_path,
                              mode=mode, project='cosmos3_av_id', group='pretrain',
                              name='eval_unit')
        assert logged['eval/overall/ate_m_rmse']==2.174
        assert logged['eval/kitti/rte_m_mean']==.043
        assert calls[-1]['mode']==mode
        assert calls[-1]['group']=='pretrain'
        assert any('eval/per_clip' in e for e in runs[-1].logs)
        assert len(runs[-1].artifacts)==1
        assert runs[-1].artifacts[0].files


def test_score_default_mode_is_disabled(monkeypatch, tmp_path):
    import cosmos3eval.cli as cli
    calls=[]
    monkeypatch.setattr(cli,'score',lambda *args,**kw: calls.append((args,kw)) or _summary())
    cli.main(['score','--eval-dir',str(tmp_path),'--pred-dir',str(tmp_path)])
    assert len(calls)==1


def test_plan_default_wandb_disabled(tmp_path, monkeypatch):
    import cosmos3av.cli as cli
    root=tmp_path/'processed'
    root.mkdir()
    rows=[]
    for dataset in ('kitti','waymo'):
        folder=root/dataset/'train'/('clip_'+dataset)
        folder.mkdir(parents=True)
        meta=folder/'metadata.json'
        meta.write_text(json.dumps({'format':'cosmos3_virtual_camera_v2',
            'pose_local_file':'poses_local_virtual_camera.npy',
            'pose_convention':'backward_framewise', 'action_normalization':'none'}))
        rows.append(dict(dataset=dataset,split='train',meta=str(meta),
                         clip_id=folder.name,video=str(folder/'video.mp4'),
                         action_file=str(folder/'actions_raw_9d.npy')))
    (root/'index.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    monkeypatch.setattr(cli,'preflight',lambda *args,**kw:{'max_dt_error_s':0.0})
    init=tmp_path/'ckpt';init.mkdir()
    vae=tmp_path/'vae';vae.write_bytes(b'dummy')
    for mode in ('disabled','offline','online'):
        out=tmp_path/mode
        args=['plan','--stage','pretrain','--datasets','kitti','waymo',
              '--processed-root',str(root), '--init-checkpoint',str(init),
              '--wan-vae',str(vae), '--outdir',str(out),
              '--max-iters','2','--save-every','2','--epoch-clips','2','--gpus','1']
        if mode!='disabled':args+=['--wandb-mode',mode]
        cli.main(args)
        assert json.loads((out/'run.json').read_text())['wandb_mode']==mode
        assert f'wandb_mode = "{mode}"' in (out/'sft_config.toml').read_text()
