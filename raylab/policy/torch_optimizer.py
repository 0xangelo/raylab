"""Standard PyTorch optimizer interface for TorchPolicies."""
import contextlib

from torch.optim import Optimizer


DEFAULT_OPTIM = "default"


class OptimizerCollection:
    """A collection of PyTorch `Optimizer`s with names."""

    def __init__(self):
        self._optimizers = {}

    def add_optimizer(self, name: str, optimizer: Optimizer):
        """Adds an optimizer to the collection.

        The optimizer can be accessed as an attribute using the given name.

        Args:
            name: the name of the optimizer
            optimizer: the optimizer instance
        """
        assert name not in self._optimizers, f"'{name}' optimizer already in collection"
        self._optimizers[name] = optimizer

    @contextlib.contextmanager
    def optimize(self, name: str = DEFAULT_OPTIM):
        """Zero grads before context and step optimizer afterwards."""
        optimizer = self._optimizers[name]
        optimizer.zero_grad()
        yield
        optimizer.step()

    def state_dict(self) -> dict:
        """Returns the state of each optimizer in the collection."""
        return {k: v.state_dict() for k, v in self._optimizers.items()}

    def load_state_dict(self, state_dict: dict):
        """Loads the state of each optimizer.

        Args:
            state_dict: optimizer collection state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        for name, optim in self._optimizers.items():
            optim.load_state_dict(state_dict[name])

    def __getattr__(self, name: str):
        if "_optimizers" in self.__dict__:
            _optimizers = self.__dict__["_optimizers"]
            if name in _optimizers:
                return _optimizers[name]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
