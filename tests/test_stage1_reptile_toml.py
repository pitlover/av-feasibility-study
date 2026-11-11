"""The [reptile] table of the Stage 1 TOML helper (CPU, no GPU)."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'scripts' / 'stage1_toml_config.py'
CONFIGS = ROOT / 'configs'

REPTILE_FIELDS = {'episode_mode', 'dataset_schedule', 'inner_steps', 'inner_warmup_steps', 'inner_grad_clip',
                  'inner_reset_optimizer', 'meta_lr', 'meta_lr_min_ratio', 'meta_lr_warmup_iters', 'clips_per_rank',
                  'support_clips', 'query_clips_per_rank', 'query_sequence_disjoint', 'eval_query_every', 'loss_mode', 'seed'}


# a minimal per_dataset plan with one unique anchor per rule under test (independent of the hand-numbered sweep files)
BASE_REPTILE_TOML = '''[job]
run_tag = "s1_reptile_test"
[paths]
edge_dcp = "edge"
wan_vae = "vae.pth"
[data.weights]
kitti = 1.0
waymo = 1.0
[train]
gpus = 4
max_iters = 100
save_every = 50
[reptile]
enabled = true
episode_mode = "per_dataset"
dataset_schedule = "alternate"
inner_steps = 10
inner_warmup_steps = 2
meta_lr = 0.5
clips_per_rank = 1
support_clips = 0
query_clips_per_rank = 1
'''


def run_helper(toml: Path, fmt: str = 'json', *overrides):
    return subprocess.run([sys.executable, str(HELPER), '--format', fmt, '--toml', str(toml), *overrides],
                          text=True, capture_output=True)


class ReptileTomlTests(unittest.TestCase):
    def test_v1_both_datasets_per_iteration(self):
        r = run_helper(CONFIGS / 'stage1_reptile_v1.toml')
        self.assertEqual(r.returncode, 0, r.stderr)
        cfg = json.loads(r.stdout)
        self.assertTrue(cfg['reptile.enabled'])
        self.assertEqual(cfg['reptile.episode_mode'], 'all_datasets')
        self.assertEqual(cfg['job.run_tag'], 's1_reptile_v1')
        self.assertEqual(cfg['train.max_iters'], 1000)
        self.assertEqual(cfg['export.step'], 1000)
        self.assertEqual(cfg['reptile.inner_steps'], 10)
        self.assertEqual(cfg['reptile.meta_lr'], 1.0)
        self.assertEqual(cfg['wandb.mode'], 'offline')
        self.assertEqual(cfg['optimizer.lr'], 1e-5)
        # hs11 budget: 1000 meta-iterations; 1000 x 2 datasets x 10 steps = 20,000 optimizer steps (~6.7x the joint baseline)
        self.assertEqual(cfg['train.max_iters'] * 2 * cfg['reptile.inner_steps'], 20000)
        self.assertEqual(cfg['train.max_iters'] % cfg['train.save_every'], 0)

    def test_variants_cover_both_episode_modes(self):
        """The sweep configs/stage1_reptile_v*.toml is numbered by hand and changes often, so only structural properties are
        checked: at least one batched (all_datasets) and one serial (per_dataset) recipe exist, a fixed support set gives every
        rank its own clips, and an alternating schedule has equal dataset weights."""
        modes: dict[str, list[str]] = {}
        for toml in sorted(CONFIGS.glob('stage1_reptile_v*.toml')):
            r = run_helper(toml)
            self.assertEqual(r.returncode, 0, f'{toml.name}: {r.stderr}')
            cfg = json.loads(r.stdout)
            modes.setdefault(cfg['reptile.episode_mode'], []).append(toml.name)
            if cfg['reptile.support_clips']:
                self.assertGreaterEqual(cfg['reptile.support_clips'], cfg['train.gpus'] * cfg['reptile.clips_per_rank'], toml.name)
            if cfg['reptile.episode_mode'] == 'per_dataset' and cfg['reptile.dataset_schedule'] == 'alternate':
                self.assertEqual(cfg['data.weights.kitti'], cfg['data.weights.waymo'], toml.name)
            self.assertEqual(cfg['reptile.query_clips_per_rank'] % cfg['reptile.clips_per_rank'], 0, toml.name)
        self.assertIn('all_datasets', modes, modes)
        self.assertIn('per_dataset', modes, modes)

    def test_all_reptile_tomls_consistent(self):
        """Every configs/stage1_reptile_*.toml parses, is a Reptile plan, and has a run_tag no other TOML uses
        (the shell refuses an existing run directory, so a copied TOML with the old run_tag cannot be planned)."""
        tomls = sorted(CONFIGS.glob('stage1_reptile_*.toml'))
        self.assertGreaterEqual(len(tomls), 3)
        run_tags = {}
        for toml in tomls:
            r = run_helper(toml)
            self.assertEqual(r.returncode, 0, f'{toml.name}: {r.stderr}')
            cfg = json.loads(r.stdout)
            self.assertTrue(cfg['reptile.enabled'], toml.name)
            self.assertEqual(cfg['export.step'], cfg['train.max_iters'], toml.name)
            self.assertEqual(cfg['train.max_iters'] % cfg['train.save_every'], 0, toml.name)
            self.assertNotIn(cfg['job.run_tag'], run_tags, f"{toml.name} reuses run_tag {cfg['job.run_tag']} of {run_tags.get(cfg['job.run_tag'])}")
            run_tags[cfg['job.run_tag']] = toml.name
        joint = json.loads(run_helper(CONFIGS / 'stage1_full.toml').stdout)
        self.assertNotIn(joint['job.run_tag'], run_tags)

    def test_smoke_toml(self):
        r = run_helper(CONFIGS / 'stage1_reptile_smoke.toml')
        self.assertEqual(r.returncode, 0, r.stderr)
        cfg = json.loads(r.stdout)
        self.assertEqual(cfg['train.max_iters'], 4)
        self.assertEqual(cfg['train.save_every'], 2)
        self.assertEqual(cfg['wandb.mode'], 'disabled')
        self.assertNotEqual(cfg['job.run_tag'], json.loads(run_helper(CONFIGS / 'stage1_reptile_v1.toml').stdout)['job.run_tag'])

    def test_joint_baseline_unchanged(self):
        r = run_helper(CONFIGS / 'stage1_full.toml')
        self.assertEqual(r.returncode, 0, r.stderr)
        cfg = json.loads(r.stdout)
        self.assertFalse(cfg['reptile.enabled'])
        self.assertEqual(cfg['job.run_tag'], 's1_pipeline_v1')
        r = run_helper(CONFIGS / 'stage1_full.toml', 'reptile-json')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('reptile.enabled is false', r.stderr)

    def test_reptile_json_payload(self):
        for toml in sorted(CONFIGS.glob('stage1_reptile_*.toml')):
            r = run_helper(toml, 'reptile-json')
            self.assertEqual(r.returncode, 0, f'{toml.name}: {r.stderr}')
            payload = json.loads(r.stdout)
            self.assertEqual(set(payload), REPTILE_FIELDS, toml.name)
            self.assertEqual(payload['seed'], 42)
        r = run_helper(CONFIGS / 'stage1_reptile_v1.toml', 'reptile-json', 'reptile.inner_steps=3', 'data.seed=7')
        payload = json.loads(r.stdout)
        self.assertEqual(payload['inner_steps'], 3)
        self.assertEqual(payload['seed'], 7)

    def test_invalid_reptile_settings(self):
        base = BASE_REPTILE_TOML
        cases = {
            'reptile.episode_mode': [('episode_mode = "per_dataset"', 'episode_mode = "bogus"')],
            'inner_warmup_steps': [('inner_warmup_steps = 2', 'inner_warmup_steps = 11')],
            'multiple of': [('\nclips_per_rank = 1\n', '\nclips_per_rank = 2\n'), ('query_clips_per_rank = 1', 'query_clips_per_rank = 3')],
            'meta_lr': [('meta_lr = 0.5', 'meta_lr = 1.5')],
            'equally': [('kitti = 1.0', 'kitti = 2.0')],
            'support_clips': [('support_clips = 0', 'support_clips = 3')],  # < gpus * clips_per_rank = 4
        }
        with tempfile.TemporaryDirectory() as d:
            for expect, replacements in cases.items():
                text = base
                for old, new in replacements:
                    self.assertEqual(text.count(old), 1, (expect, old))
                    text = text.replace(old, new, 1)
                p = Path(d) / 'bad.toml'
                p.write_text(text, encoding='utf8')
                r = run_helper(p)
                self.assertNotEqual(r.returncode, 0, expect)
                self.assertIn(expect, r.stderr, expect)
            # the same TOML with reptile disabled is accepted (the rules only apply to Reptile plans)
            p = Path(d) / 'off.toml'
            p.write_text(base.replace('enabled = true', 'enabled = false').replace('meta_lr = 0.5', 'meta_lr = 1.5'), encoding='utf8')
            r = run_helper(p)
            self.assertEqual(r.returncode, 0, r.stderr)

    def test_unknown_reptile_key_fails(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'x.toml'
            p.write_text('[reptile]\nenabled = true\nouter_lr = 0.5\n', encoding='utf8')
            r = run_helper(p)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn('Unsupported TOML key', r.stderr)

    def test_env0_carries_reptile_keys(self):
        r = subprocess.run([sys.executable, str(HELPER), '--format', 'env0', '--toml', str(CONFIGS / 'stage1_reptile_v1.toml')],
                           capture_output=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        parts = r.stdout.split(b'\0')
        pairs = dict(zip(parts[0::2], parts[1::2]))
        self.assertEqual(pairs[b'REPTILE_ENABLED'], b'true')
        self.assertEqual(pairs[b'REPTILE_EPISODE_MODE'], b'all_datasets')
        self.assertEqual(pairs[b'REPTILE_META_LR'], b'1.0')


if __name__ == '__main__':
    unittest.main()
