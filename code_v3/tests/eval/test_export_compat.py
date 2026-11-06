from pathlib import Path
import sys
import pytest
from omegaconf import OmegaConf
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cosmos3eval.export_compat import build_export_sidecar, av_metadata


def make_yaml(root: Path, *, with_loader=True):
    conf = {
        'trainer': {'memory_format': 'torch.preserve_format'},
        'model_parallel': {'params_dtype': 'torch.float32', 'autocast_dtype': 'torch.bfloat16'},
    }
    if with_loader:
        conf['dataloader_train'] = {'dataloaders': {'action_data': {'dataloader': {'dataset': {
            'list_of_datasets': [{'dataset': {
                'chunk_length': 60, 'fps': 10, 'embodiment_type': 'av',
                'manifest':'${oc.env:COSMOS_AV_MANIFEST}'
            }}]}}}}}
    else:
        conf['dataloader_train'] = {'dataloader': {}}
    OmegaConf.save(OmegaConf.create(conf), root)


def test_export_compat_non_destructive(tmp_path):
    source = tmp_path/'config.yaml'
    out = tmp_path/'export.yaml'
    make_yaml(source)
    original = source.read_bytes()
    result = build_export_sidecar(source, out, runtime_check=False)
    assert source.read_bytes() == original
    c = OmegaConf.load(out)
    assert c['_type'] == 'cosmos_framework.configs.base.config.Config'
    assert c.trainer.memory_format == 'preserve_format'
    assert c.model_parallel.params_dtype == 'float32'
    assert c.model_parallel.autocast_dtype == 'bfloat16'
    assert result == {'action_chunk_size': 60, 'conditioning_fps': 10.0, 'domain_name':'av'}


def test_missing_real_loader_is_blocked(tmp_path):
    source = tmp_path/'config.yaml'
    make_yaml(source, with_loader=False)
    with pytest.raises(ValueError, match='Saved config does not contain'):
        build_export_sidecar(source, tmp_path/'export.yaml', runtime_check=False)
    assert not (tmp_path/'export.yaml').exists()


def test_never_overwrite_original(tmp_path):
    src = tmp_path/'config.yaml'
    make_yaml(src)
    with pytest.raises(ValueError, match='Refusing'):
        build_export_sidecar(src, src, runtime_check=False)


def test_domain_mismatch_is_blocked(tmp_path):
    source = tmp_path/'config.yaml'
    make_yaml(source)
    cfg=OmegaConf.load(source)
    cfg.dataloader_train.dataloaders.action_data.dataloader.dataset.list_of_datasets[0].dataset.fps=20
    OmegaConf.save(cfg,source)
    with pytest.raises(ValueError, match='Expected AV'):
        build_export_sidecar(source, tmp_path/'export.yaml', runtime_check=False)
