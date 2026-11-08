"""Create a safe sidecar YAML for Cosmos3-Edge DCP checkpoint export.

Original model training config and DCP are never modified. NVIDIA framework
revision 98c35eda deserialize_config is checked at runtime.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
from typing import Any
from omegaconf import OmegaConf

ROOT_TYPE='cosmos_framework.configs.base.config.Config'
TORCH_ATTR_FIELDS=('trainer.memory_format','model_parallel.params_dtype','model_parallel.autocast_dtype')


def av_metadata(config:Any)->dict:
    data=OmegaConf.to_container(config,resolve=False) if OmegaConf.is_config(config) else config
    try:
        items=data['dataloader_train']['dataloaders']['action_data']['dataloader']['dataset']['list_of_datasets']
        if not items:raise ValueError('Empty AV action dataset list')
        triplets=[]
        for item in items:
            ds=item['dataset']
            triplets.append((ds['chunk_length'],ds['fps'],ds['embodiment_type']))
    except (KeyError,TypeError,IndexError) as ex:
        raise ValueError('Training config lacks canonical export-visible AV action dataset') from ex
    if any(item!=triplets[0] for item in triplets):raise ValueError('AV action dataset metadata disagree')
    steps,fps,domain=triplets[0]
    if steps!=60 or float(fps)!=10 or domain!='av':
        raise ValueError(f'Unexpected AV action contract: {triplets[0]}')
    return {'action_chunk_size':steps,'conditioning_fps':float(fps),'domain_name':domain}


def build_export_sidecar(source:Path,output:Path,*,runtime_check=True):
    source=Path(source).resolve();output=Path(output).resolve()
    if source==output:raise ValueError('Refuse to overwrite trainer config')
    if not source.is_file():raise FileNotFoundError(source)
    original=hashlib.sha256(source.read_bytes()).hexdigest()
    cfg=OmegaConf.load(source)
    meta=av_metadata(cfg)
    current=OmegaConf.select(cfg,'_type',default=None)
    if current not in (None,ROOT_TYPE):raise ValueError('Unexpected config root _type')
    if current is None:cfg['_type']=ROOT_TYPE
    import torch
    for field in TORCH_ATTR_FIELDS:
        value=OmegaConf.select(cfg,field,default=None)
        if isinstance(value,str) and value.startswith('torch.'):
            name=value.removeprefix('torch.')
            if not hasattr(torch,name):raise ValueError(f'Unknown torch attribute {field}={value}')
            OmegaConf.update(cfg,field,name)
    output.parent.mkdir(parents=True,exist_ok=True)
    tmp=output.with_name(output.name+'.tmp.yaml')
    try:
        OmegaConf.save(cfg,tmp)
        if runtime_check:
            from cosmos_framework.inference.common.config import deserialize_config
            parsed=deserialize_config(tmp)
            if not hasattr(parsed,'model') or not hasattr(parsed,'dataloader_train'):
                raise ValueError('Official config deserialization failed')
        if av_metadata(OmegaConf.load(tmp))!=meta:raise ValueError('AV dataset contract changed in sidecar')
        if hashlib.sha256(source.read_bytes()).hexdigest()!=original:
            raise RuntimeError('Source training config modified during export preparation')
        tmp.replace(output)
    finally:tmp.unlink(missing_ok=True)
    print('[EXPORT SIDECAR]',output,meta)
    return meta


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    build_export_sidecar(Path(args.input),Path(args.output))

if __name__=='__main__':main()
