"""Stage 1 Reptile meta-pretraining entry point.

    torchrun --standalone --nproc_per_node=N -m cosmos3av.entry_reptile \
        --sft-toml=<run>/sft_config.toml --reptile-config=<run>/reptile_config.json

Launched by ``cosmos3av.cli launch`` for runs whose ``run.json`` has ``trainer = "reptile"`` (created by
``cosmos3av.cli plan --reptile-config``). Registers the same ``action_inverse_dynamics_av_edge`` experiment as
the joint baseline and runs ``cosmos3av.reptile_train`` instead of ``cosmos_framework.scripts.train``.
"""
import os

# The Reptile theta copy sits next to the model's fp32 master shards; letting the CUDA allocator grow segments
# instead of fragmenting keeps head-room for the DCP save collectives (observed on 45 GiB cards). Override by
# exporting PYTORCH_CUDA_ALLOC_CONF yourself. Must be set before CUDA is initialized -> before any torch import.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from .experiment import register  # noqa: E402

if __name__ == "__main__":
    register()
    from .reptile_train import main

    main()
