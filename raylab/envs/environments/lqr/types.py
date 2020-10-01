"""Common type annotations."""
from typing import Tuple

from torch import Tensor

LQR = Tuple[Tensor, Tensor, Tensor, Tensor]
Affine = Tuple[Tensor, Tensor]
Quadratic = Tuple[Tensor, Tensor]
Box = Tuple[Tensor, Tensor]
