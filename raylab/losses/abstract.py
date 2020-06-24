"""Abstract base classes for losses."""
from abc import ABCMeta
from abc import abstractmethod
from typing import Dict
from typing import Tuple

from torch import Tensor

__all__ = [
    "Dict",
    "Tuple",
    "Tensor",
    "Loss",
]


class Loss(metaclass=ABCMeta):
    """Base interface for all loss classes.

    Attributes:
        batch_keys: the fiels in the tensor batch which will be accessed when
            called. Needed for converting the appropriate inputs to tensors
            externally.
    """

    batch_keys: Tuple[str, ...]

    @abstractmethod
    def __call__(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, float]]:
        """Computes the loss function and stats dict for the given batch.

        Subclasses should override this to implement their respective loss
        functions.

        Args:
            batch: a dictionary of input tensors

        Returns:
            A tuple with the loss tensor (not necessarily scalar) and statistics
        """

    def compile(self):
        """Optimize this loss function's implementation if supported.

        Subclasses should use TorchScript to build a static computation graph,
        although this may not be possible in some cases.
        """
