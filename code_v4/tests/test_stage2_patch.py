from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'scripts/stage2_toml_config.py'
SCRIPT = ROOT / 'stage2_full_train_eval.sh'


class TestStage2(unittest.TestCase):
    def run_parser(self, content: str, *overrides):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'config.toml'
            p.write_text(content)
            return subprocess.run([sys.executable, str(HELPER), '--toml', str(p), '--format', 'json', *overrides], capture_output=True, text=True)

    def test_explicit_stage1_paths(self):
        r=self.run_parser('''[paths]\nstage1_run="runs_stage1_full/s1_raw_v2"\nstage1_checkpoint="runs_stage1_full/s1_raw_v2/outputs/cosmos3_av_id/pretrain/s1_raw_v2/checkpoints/iter_000003000"\n[wandb]\nenabled=true\nmode="offline"\n''')
        self.assertEqual(r.returncode, 0, r.stderr)
        values=json.loads(r.stdout)
        self.assertTrue(values['paths.stage1_checkpoint'].endswith('iter_000003000'))
        self.assertEqual(values['wandb.mode'], 'offline')
        self.assertEqual(values['export.step'], values['train.max_iters'])

    def test_override(self):
        r=self.run_parser('[train]\nmax_iters=300\nsave_every=50\n', 'train.max_iters=500')
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual(json.loads(r.stdout)['train.max_iters'], 500)

    def test_disabled_default(self):
        r=self.run_parser('[job]\nrun_tag="new"\n')
        self.assertEqual(json.loads(r.stdout)['wandb.mode'], 'disabled')

    def test_unknown_key_fails(self):
        r=self.run_parser('[paths]\ns1_checkpoint="wrong"\n')
        self.assertNotEqual(r.returncode,0)
        self.assertIn('Unsupported TOML keys',r.stderr)

    def test_conflicting_wandb_fails(self):
        r=self.run_parser('[wandb]\nenabled=false\nmode="online"\n')
        self.assertNotEqual(r.returncode,0)
        self.assertIn('conflicts',r.stderr)

    def test_stage2_plan_uses_existing_stage1(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            s1=base/'runs_stage1_full/s1_raw_v2'
            ckpt=s1/'outputs/cosmos3_av_id/pretrain/s1_raw_v2/checkpoints/iter_000003000'
            (ckpt/'model').mkdir(parents=True)
            (ckpt/'trainer').mkdir(parents=True)
            (ckpt/'model/.metadata').write_text('checkpoint metadata')
            (ckpt/'trainer/.metadata').write_text('trainer metadata')
            s1.joinpath('run.json').write_text(json.dumps({'stage':'pretrain','action_frame':'camera','action_normalizer':None}))
            (base/'processed_virtual_native_v2').mkdir()
            (base/'processed_virtual_native_v2/index.jsonl').write_text('')
            (base/'cosmos-framework/cosmos_framework/scripts').mkdir(parents=True)
            (base/'cosmos-framework/cosmos_framework/scripts/train.py').write_text('#fake')
            (base/'vae.pth').write_text('vae')
            (base/'cosmos3av').mkdir()
            (base/'cosmos3av/__init__.py').write_text('')
            (base/'cosmos3av/cli.py').write_text('''
import json,sys
from pathlib import Path
args=sys.argv[1:]
assert args[0] in ('plan','smoke'), args
if args[0]=='plan':
    assert args[args.index('--stage')+1]=='posttrain'
    assert args[args.index('--init-checkpoint')+1].endswith('iter_000003000')
    out=Path(args[args.index('--outdir')+1]);out.mkdir(parents=True,exist_ok=True)
    (out/'run.json').write_text(json.dumps(dict(stage='posttrain',datasets=['av2'],action_frame='camera',action_normalizer=None,init_checkpoint=args[args.index('--init-checkpoint')+1])))
    (out/'train.jsonl').write_text('')
''')
            cfg=base/'stage2.toml'
            cfg.write_text(f'''[job]\nrun_tag="demo_s2"\n[paths]\nbase="{base}"\nstage1_run="runs_stage1_full/s1_raw_v2"\nstage1_checkpoint="runs_stage1_full/s1_raw_v2/outputs/cosmos3_av_id/pretrain/s1_raw_v2/checkpoints/iter_000003000"\nwan_vae="vae.pth"\n[train]\nmax_iters=300\nsave_every=50\n[wandb]\nenabled=true\nmode="offline"\n''')
            # install only helper in project scripts; run script from actual patch root
            # actual script's path creates a config helper from patch root
            r=subprocess.run(['bash',str(SCRIPT),'--toml',str(cfg),'plan'],capture_output=True,text=True,cwd=base)
            self.assertEqual(r.returncode,0,r.stdout+'\n'+r.stderr)
            s2=base/'runs_stage2/demo_s2'
            self.assertTrue((s2/'run.json').is_file())
            self.assertEqual(json.loads((s2/'run.json').read_text())['init_checkpoint'],str(ckpt))
            self.assertTrue((s2/'stage1_checkpoint.sha256').is_file())
            r2=subprocess.run(['bash',str(SCRIPT),'--toml',str(cfg),'train','train.max_iters=500'],capture_output=True,text=True,cwd=base)
            self.assertNotEqual(r2.returncode,0)
            self.assertIn('PLAN MISMATCH', r2.stdout+r2.stderr)

if __name__=='__main__':
    unittest.main()
