"""
Ascend Pangu Inference Framework

This package provides inference capabilities for Pangu-1B/7B models on Ascend hardware,
including quantization, pruning, and speculative inference features.
"""

__version__ = "0.1.0"

from .models.pangu_model import PanguModel
from .quantization.quantizer import Quantizer
from .pruning.pruner import Pruner
from .speculative.speculative_inference import SpeculativeInference

__all__ = [
    "PanguModel",
    "Quantizer",
    "Pruner",
    "SpeculativeInference",
]
