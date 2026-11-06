#!/usr/bin/env python3
"""CPU-only inspect the actual registered AV Edge SFT config and optional saved YAML.

Run with PYTHONPATH=<project>:<framework> and existing cosmos environment.
This does not initialize a model, train, or modify any file.
"""
from __future__ import annotations
import argparse
from pathlib import Path


def _field(node, key):
    """Read a field from either a dict/DictConfig or a Config/dataclass object."""
    from collections.abc import Mapping
    if isinstance(node, Mapping):
        return node[key]
    return getattr(node, key)


def details(node):
    # register() returns a dict-like config; the official TOML loader returns
    # cosmos_framework.configs.base.config.Config (attribute-based).
    dl = _field(node, "dataloader_train")
    datasets = _field(_field(_field(dl, "dataloaders"), "action_data"), "dataloader")
    entries = _field(_field(datasets, "dataset"), "list_of_datasets")
    dataset = _field(entries[0], "dataset")
    actual_factory = _field(dl, "_target_")
    dataset_factory = _field(dataset, "_target_")
    manifest = _field(dataset, "manifest")
    chunk_length = _field(dataset, "chunk_length")
    fps = _field(dataset, "fps")
    domain = _field(dataset, "embodiment_type")

    print("Training loader:", actual_factory)
    print("Dataset factory :", dataset_factory)
    print("Manifest        :", manifest)
    print("Action steps    :", chunk_length)
    print("FPS             :", fps)
    print("Domain          :", domain)
    assert chunk_length == 60, f"Invalid chunk length: {chunk_length!r}"
    assert float(fps) == 10.0, f"Invalid FPS: {fps!r}"
    assert domain == "av", f"Invalid domain: {domain!r}"
    assert "make_av_packing_dataloader" in str(actual_factory)
    assert "get_av_dataset_with_policy_metadata" in str(dataset_factory)
    return dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--saved-config", help="Optional config.yaml saved by NEW training run")
    p.add_argument("--sft-toml", help="Optional run/sft_config.toml; environment vars must be set")
    args = p.parse_args()
    from cosmos3av.experiment import register, _verify_contract
    print("=== REGISTERED EXPERIMENT ===")
    conf = register()
    _verify_contract(conf)
    details(conf)
    if args.sft_toml:
        from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
        print("=== RESOLVED TOML ===")
        conf = load_experiment_from_toml(Path(args.sft_toml))
        details(conf)
    if args.saved_config:
        from omegaconf import OmegaConf
        print("=== SAVED YAML ===")
        conf = OmegaConf.load(args.saved_config)
        details(conf)
    print("[OK] One canonical export-visible AV dataset configuration is present")
    print("[NOTE] This does not check the NVIDIA config serializer's _type/torch dtype compatibility")


if __name__ == "__main__":
    main()
