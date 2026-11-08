#!/usr/bin/env python3
"""Inspect the actual NVIDIA Edge AV SFT config (does not train)."""
from __future__ import annotations
import argparse
from pathlib import Path


def _field(node,key):
    from collections.abc import Mapping
    return node[key] if isinstance(node,Mapping) else getattr(node,key)


def details(node):
    dl=_field(node,'dataloader_train')
    loader=_field(_field(_field(dl,'dataloaders'),'action_data'),'dataloader')
    entries=_field(_field(loader,'dataset'),'list_of_datasets')
    dataset=_field(entries[0],'dataset')
    for key,expected in [('chunk_length',60),('fps',10),('embodiment_type','av')]:
        value=_field(dataset,key)
        print(key,value)
        if value!=expected:raise RuntimeError(f'Action contract mismatch {key}: {value}')
    if 'make_av_packing_dataloader' not in str(_field(dl,'_target_')):
        raise RuntimeError('Wrong training dataloader')
    if 'get_av_dataset_with_policy_metadata' not in str(_field(dataset,'_target_')):
        raise RuntimeError('Wrong AV dataset factory')
    return dataset


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--saved-config');parser.add_argument('--sft-toml')
    args=parser.parse_args()
    from cosmos3av.experiment import register,_verify_contract
    cfg=register();_verify_contract(cfg);details(cfg)
    if args.sft_toml:
        from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
        details(load_experiment_from_toml(Path(args.sft_toml)))
    if args.saved_config:
        from omegaconf import OmegaConf
        details(OmegaConf.load(args.saved_config))
    print('[OK] Canonical AV action dataset config inspected')

if __name__=='__main__':main()
