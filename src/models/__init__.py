"""
E-Delta Model Implementations

This module contains the core model architectures:
- BaselineGPT: Standard GPT baseline (model.py)
- DDL: Deep Delta Learning (arXiv:2601.00417)
- mHC: DeepSeek mHC with Sinkhorn (arXiv:2512.24880)
- EdeltaHybrid: E∆-MHC-Geo Hybrid (proposed model with Cayley + Householder)
"""

from .baseline_gpt import GPT as BaselineGPT, GPTConfig as BaselineConfig
from .ddl import GPT as DDLGPT, GPTConfig as DDLConfig
from .mhc import GPT as mHCGPT, GPTConfig as mHCConfig
from .edelta_hybrid import GPT as EdeltaGPT, GPTConfig as EdeltaConfig

__all__ = [
    'BaselineGPT', 'BaselineConfig',
    'DDLGPT', 'DDLConfig',
    'mHCGPT', 'mHCConfig',
    'EdeltaGPT', 'EdeltaConfig',
]
