"""Common type hints for pytorch extensions."""
from typing import Dict

from torch import Tensor

TensorDict = Dict[str, Tensor]
