"""Stubbed structural contract tests for the Cosmos3 AV export-aware recipe.

Does NOT claim an NVIDIA GPU smoke test. The real server doctor/export
still need verification against the pinned framework.
"""
import copy
import sys
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def install_mock_modules():
    def mod(name):
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    for x in [
        'cosmos3av', 'cosmos3av.framework_dataset', 'cosmos_framework',
        'cosmos_framework.utils', 'cosmos_framework.utils.lazy_config',
        'cosmos_framework.configs', 'cosmos_framework.configs.base',
        'cosmos_framework.configs.base.experiment',
        'cosmos_framework.configs.base.experiment.sft',
        'cosmos_framework.configs.base.experiment.sft.models',
        'cosmos_framework.configs.base.experiment.sft.models.edge_model_config',
        'cosmos_framework.configs.base.experiment.action',
        'cosmos_framework.configs.base.experiment.action.posttrain_config',
        'cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_libero_nano',
        'cosmos_framework.data', 'cosmos_framework.data.generator',
        'cosmos_framework.data.generator.joint_dataloader',
        'hydra', 'hydra.core', 'hydra.core.config_store',
    ]:
        mod(x)
    sys.modules['cosmos3av'].__path__ = []

    class Nemotron3DenseVLTextForCausalLM:
        pass

    def L(target):
        class Call:
            def __call__(self, **kw):
                return {'_target_': target, **kw}
        return Call()

    class Store:
        _store = {}
        @classmethod
        def instance(cls): return cls()
        def store(self, **kw):
            self._store[(kw['group'], kw['name'])] = kw['node']

    class RankPartitionedDataLoader:
        def __init__(self, datasets, **kwargs):
            self.kwargs = kwargs
            self.dataset = datasets['av']['dataset']
            self.received = datasets

    class PackingDataLoader:
        def __init__(self, dataloader, **kwargs):
            self.dataloader = dataloader
            self.kwargs = kwargs

    sys.modules['hydra.core.config_store'].ConfigStore = Store
    sys.modules['cosmos_framework.utils.lazy_config'].LazyCall = L
    sys.modules['cosmos_framework.data.generator.joint_dataloader'].RankPartitionedDataLoader = RankPartitionedDataLoader
    sys.modules['cosmos_framework.data.generator.joint_dataloader'].PackingDataLoader = PackingDataLoader

    edge = dict(
        vlm_config={'model_instance': {'_target_': Nemotron3DenseVLTextForCausalLM}},
        action_gen=True,
        diffusion_expert_config={'load_weights_from_pretrained': True},
    )
    sys.modules['cosmos_framework.configs.base.experiment.sft.models.edge_model_config'].EDGE_MODEL_CONFIG = edge

    class FakePacking:
        pass
    class FakeRank:
        pass

    lib = {
        'job': {'name': 'nano', 'group': 'libero'},
        'optimizer': {'keys_to_select': ['llm2action']},
        'model': {'config': {'vlm_config': {'model_instance': {'_target_': object}}}},
        'checkpoint': {'keys_to_skip_loading': ['net_ema.'], 'strict_resume': False, 'load_training_state': True},
        'dataloader_train': {
            '_target_': FakePacking,
            'dataset_name': 'action_libero',
            'max_samples_per_batch': 10,
            'max_sequence_length': 3000,
            'dataloader': {
                '_target_': FakeRank,
                'num_workers': 4,
                'prefetch_factor': 4,
                'datasets': {'libero': {'dataset': {'root': '${oc.env:LIBERO_ROOT}'}, 'ratio': 1}},
            },
        },
    }
    sys.modules['cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_libero_nano'].action_policy_libero_nano = lib

    def make_dataset(**kwargs):
        return ('DATASET', kwargs)
    sys.modules['cosmos3av.framework_dataset'].get_action_av_sft_dataset = make_dataset
    return Store


class TestExperiment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib.util
        cls.Store = install_mock_modules()
        spec = importlib.util.spec_from_file_location('cosmos3av.experiment', HERE.parent.parent / 'cosmos3av' / 'experiment.py')
        cls.exp = importlib.util.module_from_spec(spec)
        sys.modules['cosmos3av.experiment'] = cls.exp
        spec.loader.exec_module(cls.exp)

    def test_registration_and_single_source(self):
        cfg = self.exp.register()
        self.assertIs(cfg, self.Store._store[('experiment','action_inverse_dynamics_av_edge')])
        dl = cfg['dataloader_train']
        leaf = dl['dataloaders']['action_data']['dataloader']['dataset']['list_of_datasets'][0]['dataset']
        self.assertEqual((leaf['chunk_length'],leaf['fps'],leaf['embodiment_type']), (60,10,'av'))
        self.assertNotIn('LIBERO_ROOT', str(dl))
        self.assertIn('Nemotron', str(cfg['model']['config']['vlm_config']['model_instance']['_target_']))

        # A real config loader holds a lazy dataset node until rank-loader
        # instantiation; pass this exact node to simulated RankPartitionedDataLoader.
        built = self.exp.make_av_packing_dataloader(
            dataloaders=dl['dataloaders'],
            rank_loader_kwargs=dl['rank_loader_kwargs'],
            packing_kwargs=dl['packing_kwargs'],
        )
        self.assertIs(built.dataloader.dataset, leaf)
        self.assertEqual(built.kwargs['max_samples_per_batch'], 1)
        self.assertEqual(built.dataloader.kwargs['num_workers'], 2)
        self.assertFalse(cfg['checkpoint']['load_training_state'])

    def test_dataset_factory_validates_policy_metadata(self):
        data = self.exp.get_av_dataset_with_policy_metadata(manifest='test.jsonl',chunk_length=60,fps=10,embodiment_type='av')
        self.assertEqual(data[1]['manifest'], 'test.jsonl')
        for key, val in [('chunk_length',61), ('fps',20), ('embodiment_type','robot')]:
            with self.subTest(key=key):
                kwargs = dict(chunk_length=60,fps=10,embodiment_type='av')
                kwargs[key]=val
                with self.assertRaises(ValueError):
                    self.exp.get_av_dataset_with_policy_metadata(manifest='test.jsonl',**kwargs)

    def test_adaptor_rejects_missing_canonical_dataset(self):
        with self.assertRaises(ValueError):
            self.exp.make_av_packing_dataloader(dataloaders={},rank_loader_kwargs={},packing_kwargs={})


if __name__ == '__main__':
    unittest.main(verbosity=2)
