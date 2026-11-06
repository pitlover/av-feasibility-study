"""Compatibility bridge for Cosmos3-Edge AV training DCP -> official export.

The installed trainer's saved config.yaml can omit the inference serializer's
`_type` and include torch enum strings (e.g. torch.float32).  We NEVER alter
training config or checkpoint. We create a validated sidecar for export.

Requires cosmos-framework on PYTHONPATH to perform the real round-trip check.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


ROOT_TYPE = "cosmos_framework.configs.base.config.Config"
TORCH_ATTR_FIELDS = (
    "trainer.memory_format",
    "model_parallel.params_dtype",
    "model_parallel.autocast_dtype",
)


def av_metadata(config: Any) -> dict[str, object]:
    """Inspect *saved* dataset contract, not an invented metadata-only copy."""
    data = OmegaConf.to_container(config, resolve=False) if OmegaConf.is_config(config) else config
    try:
        loader = data["dataloader_train"]
        entries = loader["dataloaders"]["action_data"]["dataloader"]["dataset"]["list_of_datasets"]
        if not entries:
            raise ValueError("empty list_of_datasets")
        triplets = []
        for entry in entries:
            ds = entry["dataset"]
            values = (ds["chunk_length"], ds["fps"], ds["embodiment_type"])
            triplets.append(values)
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError(
            "Saved config does not contain the actual Edge AV dataset metadata at "
            "dataloader_train.dataloaders.action_data.dataloader.dataset.list_of_datasets. "
            "Do not fake this structure in a standalone export file: check the trainer first."
        ) from exc
    if any(t != triplets[0] for t in triplets[1:]):
        raise ValueError("Dataset entries disagree on chunk_length/fps/embodiment_type")
    steps, fps, domain = triplets[0]
    if (isinstance(steps, bool) or not isinstance(steps, int) or steps != 60
            or isinstance(fps, bool) or not isinstance(fps, (float, int)) or fps != 10
            or domain != "av"):
        raise ValueError(f"Expected AV 60 steps, 10 fps, 'av' domain, got {triplets[0]!r}")
    return {"action_chunk_size": steps, "conditioning_fps": float(fps), "domain_name": domain}


def build_export_sidecar(source: Path, output: Path, *, runtime_check: bool = True) -> dict[str, object]:
    """Normalize *only* known representation issues, then verify via official loader.

    Returns metadata. A failed check will not replace an existing sidecar.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Training config does not exist: {source}")
    if source == output:
        raise ValueError("Refusing to overwrite the trainer's config.yaml")
    input_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    cfg = OmegaConf.load(source)
    if not OmegaConf.is_dict(cfg):
        raise TypeError("Training config must be a YAML mapping")
    meta = av_metadata(cfg)

    current_type = OmegaConf.select(cfg, "_type", default=None)
    if current_type is not None and current_type != ROOT_TYPE:
        raise ValueError(f"Unexpected saved config _type={current_type!r}; will not overwrite")
    if current_type is None:
        cfg["_type"] = ROOT_TYPE
        print(f"[FIX] _type -> {ROOT_TYPE}", flush=True)

    import torch
    for key in TORCH_ATTR_FIELDS:
        value = OmegaConf.select(cfg, key, default=None)
        if not (isinstance(value, str) and value.startswith("torch.")):
            continue
        suffix = value.removeprefix("torch.")
        if not hasattr(torch, suffix):
            raise ValueError(f"Unrecognized torch attribute {key}={value!r}")
        OmegaConf.update(cfg, key, suffix)
        print(f"[FIX] {key}: {value} -> {suffix}", flush=True)

    # The serialized config is produced from the training config itself; no model
    # weights, loss settings, or dataset entries are invented or rewritten.
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".tmp.yaml")
    try:
        OmegaConf.save(cfg, temp)
        if runtime_check:
            from cosmos_framework.inference.common.config import deserialize_config
            restored = deserialize_config(temp)
            if not hasattr(restored, "model") or not hasattr(restored, "dataloader_train"):
                raise ValueError("Round-trip returned unexpected config type")
            print("[CHECK] Official deserialize_config: PASS", flush=True)
        meta_after = av_metadata(OmegaConf.load(temp))
        if meta != meta_after:
            raise AssertionError(f"AV action metadata changed: {meta!r} -> {meta_after!r}")
        if hashlib.sha256(source.read_bytes()).hexdigest() != input_hash:
            raise RuntimeError("Source training config changed unexpectedly; refusing export")
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)
    print(f"[CHECK] AV metadata: {meta}", flush=True)
    print(f"[OUTPUT] {output} (original config.yaml unchanged)", flush=True)
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description="Normalize and verify a Cosmos3-Edge export config (no GPU)")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    build_export_sidecar(Path(args.input), Path(args.output), runtime_check=True)


if __name__ == "__main__":
    main()
