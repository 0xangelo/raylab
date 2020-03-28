"""Distributions as PyTorch modules compatible with TorchScript."""
from .abstract import DistributionModule
from .uniform import Uniform

__all__ = ["DistributionModule", "Uniform"]
