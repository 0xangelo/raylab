"""Utilities for building optimizers."""
import contextlib

import torch
from torch.optim import Optimizer

from raylab.utils.dictionaries import all_except

from .kfac import EKFAC
from .kfac import KFAC
from .kfac import KFACMixin


OPTIMIZERS = {
    name: cls
    for name, cls in [(k, getattr(torch.optim, k)) for k in dir(torch.optim)]
    if isinstance(cls, type) and issubclass(cls, Optimizer) and cls is not Optimizer
}
OPTIMIZERS.update({"KFAC": KFAC, "EKFAC": EKFAC})


def build_optimizer(module, config):
    """Return optimizer tied to the provided module and with the desired config.

    Args:
        module (nn.Module): the module to tie the optimizer to (or its parameters)
        config (dict): mapping containing the 'type' of the optimizer and additional
            kwargs.
    """
    cls = get_optimizer_class(config["type"], wrap=True)
    if issubclass(cls, KFACMixin):
        return cls(module, **all_except(config, "type"))
    return cls(module.parameters(), **all_except(config, "type"))


def get_optimizer_class(name, wrap=True):
    """Return the optimizer class given its name.

    Arguments:
        name (str): the optimizer's name
        wrap (bool): whether to wrap the clas with `wrap_optim_cls`.

    Returns:
        The corresponding `torch.optim.Optimizer` subclass
    """
    try:
        cls = OPTIMIZERS[name]
        return wrap_optim_cls(cls) if wrap else cls
    except KeyError:
        raise ValueError(f"Couldn't find optimizer with name '{name}'")


def wrap_optim_cls(base):
    """Return PyTorch optimizer with additional context manager."""

    class ContextManagerOptim(base):
        # pylint:disable=missing-class-docstring,too-few-public-methods
        @contextlib.contextmanager
        def optimize(self):
            """Zero grads before yielding and step the optimizer upon exit."""
            self.zero_grad()
            yield
            self.step()

    return ContextManagerOptim
