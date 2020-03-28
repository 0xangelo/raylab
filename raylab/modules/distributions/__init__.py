"""Distributions as PyTorch modules compatible with TorchScript."""
from .abstract import DistributionModule, Independent
from .categorical import Categorical
from .normal import Normal
from .uniform import Uniform

__all__ = ["Categorical", "DistributionModule", "Independent", "Normal", "Uniform"]
