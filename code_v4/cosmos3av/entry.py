"""Experimental Cosmos3-Edge AV inverse dynamics run. torchrun -m cosmos3av.entry --sft-toml=<...> from Cosmos Framework root."""
import runpy
from .experiment import register

if __name__ == '__main__':
    register()
    runpy.run_module('cosmos_framework.scripts.train', run_name='__main__', alter_sys=True)
