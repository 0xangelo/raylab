"""Collection of type annotations."""
from typing import Callable
from typing import Dict
from typing import Tuple
from typing import Union

from torch import Tensor


DynamicsFn = Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]

RewardFn = Callable[[Tensor, Tensor, Tensor], Tensor]

StatDict = Dict[str, Union[float, int]]

TensorDict = Dict[str, Tensor]

TerminationFn = Callable[[Tensor, Tensor, Tensor], Tensor]
