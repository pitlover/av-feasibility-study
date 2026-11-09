"""CLI-only tests; does not launch GPU jobs."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'scripts' / 'stage1_toml_config.py'
SHELL = ROOT / 'stage1_full_train_eval.sh'
CONFIG = ROOT / 'configs' / 'stage1_full.toml'


class TestStage1Toml(unittest.TestCase):
    def run_helper(self, *params, env=None):
        return subprocess.run([sys.executable, str(HELPER), '--format', 'json', '--toml', str(CONFIG), *params],
                              text=True, capture_output=True, env=env)

    def test_sample(self):
        r = self.run_helper()
        self.assertEqual(r.returncode, 0, r.stderr)
        obj = json.loads(r.stdout)
        self.assertEqual(obj['train.max_iters'], 3000)
        self.assertEqual(obj['export.step'], 3000)
        self.assertEqual(obj['wandb.mode'], 'disabled')
        self.assertEqual(obj['data.weights.kitti'], 1.0)

    def test_precedence_and_override(self):
        r = self.run_helper('train.max_iters=500', 'train.save_every=100', 'wandb.enabled=true', 'wandb.mode=offline')
        self.assertEqual(r.returncode, 0, r.stderr)
        obj = json.loads(r.stdout)
        self.assertEqual(obj['train.max_iters'], 500)
        self.assertEqual(obj['export.step'], 500)  # step=0 follows the override
        self.assertEqual(obj['wandb.mode'], 'offline')

    def test_unknown_keys_fail(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'a.toml'
            p.write_text('[train]\nbatch_size=64\n')
            r = subprocess.run([sys.executable, str(HELPER), '--format', 'json', '--toml', str(p)],
                               capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn('Unsupported TOML key', r.stderr)

    def test_bad_wandb_fails(self):
        r = self.run_helper('wandb.enabled=false', 'wandb.mode=online')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('conflicts', r.stderr)

    def test_shell_config_command(self):
        r = subprocess.run(['bash', str(SHELL), 'config', '--toml', str(CONFIG)],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('s1_toml_v1', r.stdout)

    def test_disabled_no_wandb_dependency(self):
        r = self.run_helper()
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == '__main__':
    unittest.main()
