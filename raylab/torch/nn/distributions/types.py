# pylint:disable=missing-module-docstring
from typing import Dict

from torch import Tensor

__all__ = ["DistParams"]

DistParams = Dict[str, Tensor]
