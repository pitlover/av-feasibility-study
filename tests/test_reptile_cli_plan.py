"""Real `cosmos3av.cli plan --reptile-config` / `launch` (dry run) on the bundled sample clips (CPU, no framework, no GPU)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

import reptile_testutil as tu

ROOT = tu.ROOT
HELPER = ROOT / 'scripts' / 'stage1_toml_config.py'


def _env():
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    return env


@unittest.skipUnless(tu.have_sample_clips(), 'bundled sample clips not available')
class PlanWithReptileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.base = base
        self.index = tu.write_local_index(base / 'processed')
        (base / 'edge').mkdir()
        (base / 'vae.pth').touch()
        (base / 'fw/cosmos_framework/scripts').mkdir(parents=True)
        (base / 'fw/cosmos_framework/scripts/train.py').write_text('')
        r = subprocess.run([sys.executable, str(HELPER), '--format', 'reptile-json', '--toml',
                            str(ROOT / 'configs/stage1_reptile_v1.toml'), 'reptile.episode_mode=per_dataset',
                            'reptile.inner_steps=2', 'train.gpus=2'],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.reptile_json = base / 'reptile.json'
        self.reptile_json.write_text(r.stdout)

    def tearDown(self):
        self.tmp.cleanup()

    def plan(self, outdir: Path, *extra, stage='pretrain', weights='kitti:1,waymo:1'):
        datasets = ['kitti', 'waymo'] if stage == 'pretrain' else ['av2']
        cmd = [sys.executable, '-m', 'cosmos3av.cli', 'plan', '--stage', stage, '--datasets', *datasets, '--weights', weights,
               '--processed-root', str(self.index.parent), '--init-checkpoint', str(self.base / 'edge'), '--wan-vae',
               str(self.base / 'vae.pth'), '--outdir', str(outdir), '--epoch-clips', '12', '--max-iters', '4',
               '--save-every', '2', '--lr', '1e-5', '--gpus', '2', '--seed', '42', '--wandb-mode', 'offline',
               '--full-preflight', *extra]
        return subprocess.run(cmd, cwd=ROOT, env=_env(), text=True, capture_output=True)

    def test_plan_writes_reptile_run(self):
        run_dir = self.base / 'runs/s1_reptile_test'
        r = self.plan(run_dir, '--reptile-config', str(self.reptile_json))
        self.assertEqual(r.returncode, 0, r.stdout + '\n' + r.stderr)
        self.assertIn('[reptile] mode=per_dataset', r.stdout)
        detail = json.loads((run_dir / 'run.json').read_text())
        self.assertEqual(detail['trainer'], 'reptile')
        self.assertEqual(detail['stage'], 'pretrain')
        self.assertEqual(detail['action_frame'], 'camera')
        self.assertIsNone(detail['action_normalizer'])
        self.assertEqual(Path(detail['reptile_config']), run_dir / 'reptile_config.json')
        self.assertEqual(detail['reptile_clips']['kitti']['clips'], len({r['clip_id'] for r in tu.sample_rows() if r['dataset'] == 'kitti' and r['split'] == 'train'}))
        self.assertEqual(detail['reptile_support_clips_per_task'], 2 * 2 * 1)
        self.assertEqual(detail['reptile_query_clips_per_task'], 2)
        cfg = json.loads((run_dir / 'reptile_config.json').read_text())
        self.assertEqual(cfg['datasets'], ['kitti', 'waymo'])
        self.assertEqual(cfg['dataset_weights'], {'kitti': 1.0, 'waymo': 1.0})
        self.assertEqual(cfg['inner_steps'], 2)
        self.assertEqual(Path(cfg['manifest']), run_dir / 'train_unique.jsonl')
        unique = [json.loads(l) for l in (run_dir / 'train_unique.jsonl').read_text().splitlines() if l.strip()]
        ids = [u['clip_id'] for u in unique]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(len(ids), len(set(ids)))
        train_rows = [json.loads(l) for l in (run_dir / 'train.jsonl').read_text().splitlines() if l.strip()]
        self.assertEqual(len(train_rows), 12)  # the canonical (repeating) manifest is still written
        self.assertEqual(set(ids), {t['clip_id'] for t in train_rows} | set(ids))
        toml = tomllib.loads((run_dir / 'sft_config.toml').read_text())
        self.assertFalse(toml['model']['ema']['enabled'])
        self.assertEqual(toml['trainer']['max_iter'], 4)
        self.assertEqual(toml['job']['wandb_mode'], 'offline')
        self.assertEqual(toml['job']['experiment'], 'action_inverse_dynamics_av_edge')

        # launch (dry run) routes to the Reptile entry point
        l = subprocess.run([sys.executable, '-m', 'cosmos3av.cli', 'launch', '--run-dir', str(run_dir), '--framework-root',
                            str(self.base / 'fw')], cwd=ROOT, env=_env(), text=True, capture_output=True)
        self.assertEqual(l.returncode, 0, l.stdout + '\n' + l.stderr)
        self.assertIn('[TRAINER] reptile', l.stdout)
        self.assertIn('cosmos3av.entry_reptile', l.stdout)
        self.assertIn(f'--reptile-config={run_dir / "reptile_config.json"}', l.stdout)
        self.assertIn('--nproc_per_node=2', l.stdout)
        self.assertIn('[DRY RUN]', l.stdout)

    def test_plan_without_reptile_is_unchanged(self):
        run_dir = self.base / 'runs/s1_joint_test'
        r = self.plan(run_dir)
        self.assertEqual(r.returncode, 0, r.stdout + '\n' + r.stderr)
        detail = json.loads((run_dir / 'run.json').read_text())
        self.assertEqual(detail['trainer'], 'sft')
        self.assertNotIn('reptile_config', detail)
        self.assertFalse((run_dir / 'reptile_config.json').exists())
        self.assertNotIn('[model.ema]', (run_dir / 'sft_config.toml').read_text())
        l = subprocess.run([sys.executable, '-m', 'cosmos3av.cli', 'launch', '--run-dir', str(run_dir), '--framework-root',
                            str(self.base / 'fw')], cwd=ROOT, env=_env(), text=True, capture_output=True)
        self.assertEqual(l.returncode, 0, l.stdout + '\n' + l.stderr)
        self.assertIn('[TRAINER] sft', l.stdout)
        self.assertIn('-m cosmos3av.entry ', l.stdout + ' ')
        self.assertNotIn('entry_reptile', l.stdout)

    def test_reptile_rejected_for_stage2_and_bad_weights(self):
        r = self.plan(self.base / 'runs/bad_stage', '--reptile-config', str(self.reptile_json), stage='posttrain', weights='av2:1')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('Stage 1', r.stderr)
        self.assertFalse((self.base / 'runs/bad_stage/run.json').exists())
        r = self.plan(self.base / 'runs/bad_weights', '--reptile-config', str(self.reptile_json), weights='kitti:2,waymo:1')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("alternate", r.stderr)
        self.assertFalse((self.base / 'runs/bad_weights/run.json').exists())


if __name__ == '__main__':
    unittest.main()
