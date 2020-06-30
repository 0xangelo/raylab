"""Collection of type annotations."""
from typing import Callable
from typing import Dict
from typing import Tuple

from torch import Tensor

TensorDict = Dict[str, Tensor]
RewardFn = Callable[[Tensor, Tensor, Tensor], Tensor]
TerminationFn = Callable[[Tensor, Tensor, Tensor], Tensor]
DynamicsFn = Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]
DetPolicy = Callable[[Tensor], Tensor]
StochasticPolicy = Callable[[Tensor], Tuple[Tensor, Tensor]]
ActionValue = Callable[[Tensor, Tensor], Tensor]
StateValue = Callable[[Tensor], Tensor]
