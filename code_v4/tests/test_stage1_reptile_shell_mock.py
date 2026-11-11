"""stage1_full_train_eval.sh with a Reptile TOML against a mocked cosmos3av.cli (CPU, no GPU)."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHELL = ROOT / 'stage1_full_train_eval.sh'

FAKE_PYTHON = '''#!/usr/bin/env python3
import os,sys,json,re,shutil
from pathlib import Path
args=sys.argv[1:]
if args[:2]==['-m','cosmos3av.cli']:
    mode=args[2]
    if mode=='plan':
        root=Path(args[args.index('--outdir')+1]); root.mkdir(parents=True)
        (root/'train.jsonl').write_text('')
        (root/'val.jsonl').write_text('')
        iters=int(args[args.index('--max-iters')+1])
        detail={'stage':'pretrain','action_frame':'camera','action_normalizer':None,'n_train':12,'n_val':4,
                'weights':args[args.index('--weights')+1],'wandb_mode':args[args.index('--wandb-mode')+1],
                'gpus':int(args[args.index('--gpus')+1]),'trainer':'sft'}
        toml='[trainer]\\nmax_iter = %d\\n' % iters
        if '--reptile-config' in args:
            src=Path(args[args.index('--reptile-config')+1])
            cfg=json.loads(src.read_text())
            dst=root/'reptile_config.json'; dst.write_text(json.dumps(cfg))
            detail.update(trainer='reptile', reptile_config=str(dst), reptile_clips={'kitti':{'clips':3},'waymo':{'clips':3}})
            toml+='[model.ema]\\nenabled = false\\n'
        (root/'sft_config.toml').write_text(toml)
        (root/'run.json').write_text(json.dumps(detail))
        sys.exit(0)
    if mode=='smoke': sys.exit(0)
    if mode=='launch':
        root=Path(args[args.index('--run-dir')+1]); n=root.name
        detail=json.loads((root/'run.json').read_text())
        iters=int(re.search(r'max_iter = (\\d+)',(root/'sft_config.toml').read_text()).group(1))
        d=root/'outputs/cosmos3_av_id/pretrain'/n/('checkpoints/iter_%09d' % iters)
        (d/'model').mkdir(parents=True); (d/'trainer').mkdir()
        (d/'model/.metadata').touch(); (d/'trainer/.metadata').touch()
        (root/'launched_trainer.txt').write_text(detail.get('trainer','sft'))
        sys.exit(0)
os.execv(sys.executable,[sys.executable,*args])
'''

REPTILE_TOML = '''[job]
run_tag="mock_reptile"
[paths]
edge_dcp="edge"
wan_vae="vae.pth"
[data]
seed=11
[data.weights]
kitti=1
waymo=1
[train]
gpus=2
max_iters=4
save_every=2
[wandb]
enabled=true
mode="offline"
[reptile]
enabled=true
episode_mode="per_dataset"
dataset_schedule="alternate"
inner_steps=3
inner_warmup_steps=1
meta_lr=0.5
'''


class ReptileShellMockTests(unittest.TestCase):
    def test_plan_train_and_frozen_reptile_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / 'bin').mkdir()
            (base / 'processed_virtual_native_v2').mkdir()
            (base / 'processed_virtual_native_v2/index.jsonl').write_text('')
            (base / 'cosmos-framework/cosmos_framework/scripts').mkdir(parents=True)
            (base / 'cosmos-framework/cosmos_framework/scripts/train.py').write_text('')
            (base / 'edge').mkdir()
            (base / 'vae.pth').touch()
            py = base / 'bin/python'
            py.write_text(FAKE_PYTHON)
            py.chmod(0o755)
            config = base / 'stage1_reptile.toml'
            config.write_text(REPTILE_TOML)
            env = os.environ.copy()
            env.update(PATH=f"{base / 'bin'}:{env['PATH']}", BASE=str(base))

            def run(mode, *extra):
                return subprocess.run(['bash', str(SHELL), '--toml', str(config), mode, *extra], cwd=base, env=env,
                                      text=True, capture_output=True)

            c = run('config')
            self.assertEqual(c.returncode, 0, c.stdout + c.stderr)
            self.assertEqual(json.loads(c.stdout[c.stdout.index('{'):])['reptile.inner_steps'], 3)

            p = run('plan')
            self.assertEqual(p.returncode, 0, p.stdout + '\n' + p.stderr)
            self.assertIn('trainer=reptile', p.stdout)
            self.assertIn('[PLAN] Reptile meta-pretraining', p.stdout)
            self.assertIn('[PLAN OK] reptile config:', p.stdout)
            run_dir = base / 'runs_stage1_full/mock_reptile'
            report = json.loads((run_dir / 'run.json').read_text())
            self.assertEqual(report['trainer'], 'reptile')
            self.assertEqual(report['wandb_mode'], 'offline')
            rcfg = json.loads((run_dir / 'reptile_config.json').read_text())
            self.assertEqual(rcfg['episode_mode'], 'per_dataset')
            self.assertEqual(rcfg['inner_steps'], 3)
            self.assertEqual(rcfg['meta_lr'], 0.5)
            self.assertEqual(rcfg['seed'], 11)
            self.assertNotIn('enabled', rcfg)
            self.assertIn('reptile.inner_steps', (run_dir / 'resolved_stage1_config.json').read_text())
            # the temp JSON handed to the planner is cleaned up
            self.assertFalse(any(x.name.startswith('tmp') for x in base.iterdir() if x.is_file()))

            q = run('train')
            self.assertEqual(q.returncode, 0, q.stdout + '\n' + q.stderr)
            self.assertTrue((run_dir / 'outputs/cosmos3_av_id/pretrain/mock_reptile/checkpoints/iter_000000004/model/.metadata').exists())
            self.assertEqual((run_dir / 'launched_trainer.txt').read_text(), 'reptile')

            # a Reptile knob changed after plan -> refuse to train
            (run_dir / 'train.log').unlink()
            config.write_text(REPTILE_TOML.replace('inner_steps=3', 'inner_steps=5'))
            r = run('train')
            self.assertNotEqual(r.returncode, 0)
            self.assertIn('[CONFIG MISMATCH] reptile.inner_steps', r.stdout + r.stderr)

    def test_joint_plan_still_sft(self):
        """A TOML without [reptile] plans the joint baseline exactly as before (no --reptile-config, trainer=sft)."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / 'bin').mkdir()
            (base / 'processed_virtual_native_v2').mkdir()
            (base / 'processed_virtual_native_v2/index.jsonl').write_text('')
            (base / 'cosmos-framework/cosmos_framework/scripts').mkdir(parents=True)
            (base / 'cosmos-framework/cosmos_framework/scripts/train.py').write_text('')
            (base / 'edge').mkdir()
            (base / 'vae.pth').touch()
            py = base / 'bin/python'
            py.write_text(FAKE_PYTHON)
            py.chmod(0o755)
            config = base / 'stage1.toml'
            config.write_text('[job]\nrun_tag="mock_sft"\n[paths]\nedge_dcp="edge"\nwan_vae="vae.pth"\n[train]\nmax_iters=6\nsave_every=3\n')
            env = os.environ.copy()
            env.update(PATH=f"{base / 'bin'}:{env['PATH']}", BASE=str(base))
            p = subprocess.run(['bash', str(SHELL), '--toml', str(config), 'plan'], cwd=base, env=env, text=True, capture_output=True)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertIn('trainer=sft', p.stdout)
            self.assertNotIn('Reptile', p.stdout)
            report = json.loads((base / 'runs_stage1_full/mock_sft/run.json').read_text())
            self.assertEqual(report['trainer'], 'sft')
            self.assertFalse((base / 'runs_stage1_full/mock_sft/reptile_config.json').exists())


if __name__ == '__main__':
    unittest.main()
