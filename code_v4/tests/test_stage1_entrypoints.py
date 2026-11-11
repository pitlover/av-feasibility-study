"""No-argument entrypoints configs/scripts/run_stage1_reptile_*.sh, [job] run_step and the [paths] output roots (CPU)."""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / 'stage1_full_train_eval.sh'
WRAPPERS = ROOT / 'configs' / 'scripts'
CONFIGS = ROOT / 'configs'


def run(cmd, cwd=ROOT, env=None):
    return subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)


def parse_config_output(stdout: str) -> dict:
    """The `config` step prints [OUTPUT]/[TOML] lines followed by the resolved JSON."""
    return json.loads(stdout[stdout.index('{'):])


class EntrypointTests(unittest.TestCase):
    def test_every_wrapper_names_an_existing_toml_and_delegates(self):
        wrappers = sorted(WRAPPERS.glob('run_stage1_reptile_*.sh'))
        self.assertGreaterEqual(len(wrappers), 2)
        for w in wrappers:
            text = w.read_text(encoding='utf8')
            m = re.search(r'(?m)^TOML_FILE="configs/(stage1_reptile_[^"]+\.toml)"$', text)
            self.assertIsNotNone(m, w.name)
            self.assertTrue((CONFIGS / m.group(1)).is_file(), f'{w.name}: {m.group(1)} missing')
            self.assertEqual(w.name, m.group(1).replace('stage1_', 'run_stage1_').replace('.toml', '.sh'))
            self.assertIn('exec bash stage1_full_train_eval.sh --toml "$TOML_FILE"', text, w.name)
            self.assertEqual(run(['bash', '-n', str(w)]).returncode, 0, w.name)

    def test_wrapper_without_run_step_prints_config_of_its_toml(self):
        """A TOML without [job] run_step resolves to the harmless `config` step: the wrapper must find the project
        root from configs/scripts/, cd there and print the resolved settings of ITS TOML (no training side effects)."""
        for w in sorted(WRAPPERS.glob('run_stage1_reptile_v*.sh')):
            toml = CONFIGS / re.search(r'TOML_FILE="configs/([^"]+)"', w.read_text(encoding='utf8')).group(1)
            helper = json.loads(run(['python', str(ROOT / 'scripts' / 'stage1_toml_config.py'), '--format', 'json',
                                     '--toml', str(toml)]).stdout)
            if helper['job.run_step'] != 'config':
                continue  # a TOML that launches work (e.g. the smoke pipeline) is not exercised on CPU
            r = run(['bash', str(w)], cwd=tempfile.gettempdir())  # cwd deliberately elsewhere
            self.assertEqual(r.returncode, 0, f'{w.name}: {r.stderr}')
            self.assertIn(f'[STAGE1] TOML={ROOT / "configs" / toml.name}', r.stdout, w.name)
            cfg = parse_config_output(r.stdout)
            self.assertEqual(cfg['job.run_tag'], helper['job.run_tag'], w.name)
            self.assertTrue(cfg['reptile.enabled'], w.name)
            self.assertIn(f"[OUTPUT] stage1_run={ROOT}/runs_stage1_full/{cfg['job.run_tag']}", r.stdout, w.name)

    def test_wrapper_rejects_arguments(self):
        w = WRAPPERS / 'run_stage1_reptile_v1.sh'
        r = run(['bash', str(w), 'plan'])
        self.assertEqual(r.returncode, 2)
        self.assertIn('takes no arguments', r.stderr)

    def test_wrapper_copied_to_project_root_still_works(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ('scripts', 'configs'):
                os.symlink(ROOT / name, root / name)
            os.symlink(DRIVER, root / 'stage1_full_train_eval.sh')
            w = root / 'run_stage1_reptile_v1.sh'
            w.write_text((WRAPPERS / 'run_stage1_reptile_v1.sh').read_text(encoding='utf8'), encoding='utf8')
            r = run(['bash', str(w)], cwd=root)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(parse_config_output(r.stdout)['job.run_tag'], 's1_reptile_v1')

    def test_driver_uses_job_run_step_when_no_step_is_given(self):
        with tempfile.TemporaryDirectory() as d:
            toml = Path(d) / 'x.toml'
            toml.write_text('[job]\nrun_tag = "s1_runstep_test"\nrun_step = "config"\n', encoding='utf8')
            r = run(['bash', str(DRIVER), '--toml', str(toml)])
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(parse_config_output(r.stdout)['job.run_step'], 'config')
            # an explicit step on the command line wins over the TOML
            toml.write_text('[job]\nrun_tag = "s1_runstep_test"\nrun_step = "all"\n', encoding='utf8')
            r = run(['bash', str(DRIVER), '--toml', str(toml), 'config'])
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(parse_config_output(r.stdout)['job.run_step'], 'all')
            # invalid step names are rejected by the helper before anything runs
            toml.write_text('[job]\nrun_step = "launch"\n', encoding='utf8')
            r = run(['bash', str(DRIVER), '--toml', str(toml)])
            self.assertNotEqual(r.returncode, 0)
            self.assertIn('job.run_step must be one of', r.stderr)

    def test_output_roots_default_and_override(self):
        base = run(['bash', str(DRIVER), '--toml', str(CONFIGS / 'stage1_reptile_v1.toml'), 'config']).stdout
        self.assertIn(f'[OUTPUT] stage1_run={ROOT}/runs_stage1_full/s1_reptile_v1', base)
        self.assertIn(f'[OUTPUT] final_dcp={ROOT}/runs_stage1_full/s1_reptile_v1/outputs/cosmos3_av_id/pretrain/'
                      f's1_reptile_v1/checkpoints/iter_000001000', base)
        self.assertIn(f'[OUTPUT] export={ROOT}/exports/s1_reptile_v1_iter_1000', base)
        self.assertIn(f'[OUTPUT] pred_dir={ROOT}/evaluation/pred_s1_reptile_v1_iter_1000_val_seed42_v1', base)
        over = run(['bash', str(DRIVER), '--toml', str(CONFIGS / 'stage1_reptile_v1.toml'), 'config',
                    'paths.runs_root=/scratch/runs', 'paths.exports_root=my_exports', 'paths.evaluation_root=/scratch/eval/']).stdout
        self.assertIn('[OUTPUT] stage1_run=/scratch/runs/s1_reptile_v1', over)
        self.assertIn(f'[OUTPUT] export={ROOT}/my_exports/s1_reptile_v1_iter_1000', over)
        self.assertIn('[OUTPUT] eval_dir=/scratch/eval/stage1_s1_reptile_v1/kitti_waymo_val_seed42_v1', over)
        cfg = parse_config_output(over)
        self.assertEqual(cfg['paths.runs_root'], '/scratch/runs')
        r = run(['bash', str(DRIVER), '--toml', str(CONFIGS / 'stage1_reptile_v1.toml'), 'config', 'paths.runs_root=/'])
        self.assertNotEqual(r.returncode, 0)


if __name__ == '__main__':
    unittest.main()
