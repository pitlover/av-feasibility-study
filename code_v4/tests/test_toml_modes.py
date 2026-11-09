"""Basic CPU-only configuration tests; run `python -m unittest discover -s tests`."""
from pathlib import Path
import importlib.util
import json
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

def load(filename):
    path = ROOT / 'scripts' / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class UnifiedConfigTests(unittest.TestCase):
    def test_stage1_modes_and_paths(self):
        parser = load('stage1_toml_config.py')
        source = ROOT / 'configs' / 'stage1_full.toml'
        for mode in ('score', 'render', 'both', 'full'):
            cfg = parser.read_config(str(source), [f'eval.mode={mode}'])
            self.assertEqual(cfg['eval.mode'], mode)
        cfg = parser.read_config(str(source), [
            'eval.pred_dir=evaluation/custom_retry1',
            'eval.eval_dir=evaluation/custom_selected',
        ])
        self.assertEqual(cfg['eval.pred_dir'], 'evaluation/custom_retry1')
        self.assertEqual(cfg['eval.eval_dir'], 'evaluation/custom_selected')

    def test_stage2_links_correct_stage1(self):
        stage1 = load('stage1_toml_config.py').read_config(str(ROOT / 'configs' / 'stage1_full.toml'), [])
        stage2 = load('stage2_toml_config.py').resolve(str(ROOT / 'configs' / 'stage2_full.toml'), [])
        tag = stage1['job.run_tag']
        self.assertEqual(stage2['paths.stage1_run'], f'runs_stage1_full/{tag}')
        self.assertIn(f'/pretrain/{tag}/checkpoints/iter_{stage1["train.max_iters"]:09d}', stage2['paths.stage1_checkpoint'])
        self.assertEqual(stage2['wandb.mode'], 'disabled')

    def test_stage2_modes_and_override(self):
        parser = load('stage2_toml_config.py')
        source = ROOT / 'configs' / 'stage2_full.toml'
        for mode in ('score', 'render', 'both', 'full'):
            cfg = parser.resolve(str(source), [f'eval.mode={mode}'])
            self.assertEqual(cfg['eval.mode'], mode)
        cfg=parser.resolve(str(source), ['eval.pred_dir=evaluation/retry1','wandb.enabled=true','wandb.mode=offline'])
        self.assertEqual(cfg['eval.pred_dir'], 'evaluation/retry1')
        self.assertEqual(cfg['wandb.mode'], 'offline')

    def test_invalid_mode_fails(self):
        for parser_name, func in [('stage1_toml_config.py','read_config'), ('stage2_toml_config.py', 'resolve')]:
            p=load(parser_name)
            with self.assertRaises(ValueError):
                getattr(p, func)(str(ROOT / 'configs' / ('stage1_full.toml' if func=='read_config' else 'stage2_full.toml')), ['eval.mode=unknown'])

if __name__ == '__main__':
    unittest.main()
