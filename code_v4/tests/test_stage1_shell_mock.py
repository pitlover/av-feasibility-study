import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SHELL=ROOT/'stage1_full_train_eval.sh'


class ShellMockTests(unittest.TestCase):
    def test_plan_and_train_cli_integration(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            (base/'bin').mkdir()
            (base/'processed_virtual_native_v2').mkdir()
            (base/'processed_virtual_native_v2/index.jsonl').write_text('')
            (base/'cosmos-framework/cosmos_framework/scripts').mkdir(parents=True)
            (base/'cosmos-framework/cosmos_framework/scripts/train.py').write_text('')
            (base/'edge').mkdir()
            (base/'vae.pth').touch()
            (base/'runs_virtual/stage1_joint_300').mkdir(parents=True)
            (base/'runs_virtual/stage1_joint_300/run.json').write_text(json.dumps({
                'init_checkpoint':str(base/'edge'), 'wan_vae':str(base/'vae.pth')}))
            # Python wrapper delegates to real python except Cosmos3 training calls.
            py=base/'bin/python'
            py.write_text('''#!/usr/bin/env python3
import os,sys,json,subprocess
from pathlib import Path
args=sys.argv[1:]
if args[:2]==['-m','cosmos3av.cli']:
    mode=args[2]
    if mode=='plan':
        root=Path(args[args.index('--outdir')+1]); root.mkdir(parents=True)
        (root/'train.jsonl').write_text('')
        (root/'val.jsonl').write_text('')
        (root/'sft_config.toml').write_text('[trainer]\\nmax_iter = 3000\\n')
        (root/'run.json').write_text(json.dumps({'stage':'pretrain','action_frame':'camera',
            'action_normalizer':None,'n_train':12000,'n_val':40,'weights':args[args.index('--weights')+1],
            'wandb_mode':args[args.index('--wandb-mode')+1]}))
        sys.exit(0)
    if mode=='smoke': sys.exit(0)
    if mode=='launch':
        root=Path(args[args.index('--run-dir')+1]); n=root.name
        d=root/'outputs/cosmos3_av_id/pretrain'/n/'checkpoints/iter_000003000'
        (d/'model').mkdir(parents=True); (d/'trainer').mkdir()
        (d/'model/.metadata').touch(); (d/'trainer/.metadata').touch()
        sys.exit(0)
os.execv(sys.executable,[sys.executable,*args])
''')
            py.chmod(0o755)
            config=base/'stage1.toml'
            config.write_text('''[job]
run_tag="mock_run"
[paths]
edge_dcp="edge"
wan_vae="vae.pth"
[data.weights]
kitti=2
waymo=1
[wandb]
enabled=true
mode="offline"
''')
            env=os.environ.copy()
            env.update(PATH=f"{base/'bin'}:{env['PATH']}", BASE=str(base))
            def run(mode):
                return subprocess.run(['bash',str(SHELL),'--toml',str(config),mode],cwd=base,env=env,text=True,capture_output=True)
            p=run('plan')
            self.assertEqual(p.returncode,0, p.stdout+'\n'+p.stderr)
            report=json.loads((base/'runs_stage1_full/mock_run/run.json').read_text())
            self.assertEqual(report['weights'],'kitti:2.0,waymo:1.0')
            self.assertEqual(report['wandb_mode'],'offline')
            q=run('train')
            self.assertEqual(q.returncode,0, q.stdout+'\n'+q.stderr)
            self.assertTrue((base/'runs_stage1_full/mock_run/outputs/cosmos3_av_id/pretrain/mock_run/checkpoints/iter_000003000/model/.metadata').exists())
            config.write_text(config.read_text().replace('kitti=2','kitti=3'))
            r=run('train')
            self.assertNotEqual(r.returncode,0)
            self.assertIn('[CONFIG MISMATCH]',r.stdout+r.stderr)

if __name__=='__main__': unittest.main()
