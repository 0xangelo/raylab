"""Collection of type annotations."""
from typing import Callable
from typing import Tuple

from torch import Tensor

RewardFn = Callable[[Tensor, Tensor, Tensor], Tensor]
TerminationFn = Callable[[Tensor, Tensor, Tensor], Tensor]
DynamicsFn = Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]
DetPolicy = Callable[[Tensor], Tensor]
ActionValue = Callable[[Tensor, Tensor], Tensor]
StateValue = Callable[[Tensor], Tensor]
