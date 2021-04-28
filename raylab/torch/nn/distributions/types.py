# pylint:disable=missing-module-docstring
from typing import Dict, Tuple

from torch import Tensor

__all__ = ["DistParams", "SampleLogp"]

DistParams = Dict[str, Tensor]
SampleLogp = Tuple[Tensor, Tensor]
