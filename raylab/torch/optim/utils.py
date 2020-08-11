"""Utilities for building optimizers."""
import contextlib
from typing import Type

import torch
import torch.nn as nn
from torch.optim import Optimizer

from raylab.utils.dictionaries import all_except

from .kfac import EKFAC
from .kfac import KFAC
from .radam import AdamW
from .radam import PlainRAdam
from .radam import RAdam


OPTIMIZERS = {
    name: cls
    for name, cls in [(k, getattr(torch.optim, k)) for k in dir(torch.optim)]
    if isinstance(cls, type) and issubclass(cls, Optimizer) and cls is not Optimizer
}
OPTIMIZERS.update(
    {
        "KFAC": KFAC,
        "EKFAC": EKFAC,
        "RAdam": RAdam,
        "PlainRAdam": PlainRAdam,
        "AdamW": AdamW,
    }
)


def build_optimizer(module: nn.Module, config: dict, wrap: bool = False) -> Optimizer:
    """Build optimizer with the desired config and tied to a module.

    Args:
        module: the module to tie the optimizer to (or its parameters)
        config: mapping containing the 'type' of the optimizer and additional
            kwargs.
        wrap: whether to wrap the class with :func:`wrap_optim_cls`.
    """
    cls = get_optimizer_class(config["type"], wrap=wrap)
    return link_optimizer(cls, module, all_except(config, "type"))


def get_optimizer_class(name: str, wrap: bool = True) -> Type[Optimizer]:
    """Return the optimizer class given its name.

    Args:
        name: the optimizer's name
        wrap: whether to wrap the class with :func:`wrap_optim_cls`.

    Returns:
        The optimizer class
    """
    try:
        cls = OPTIMIZERS[name]
        return wrap_optim_cls(cls) if wrap else cls
    except KeyError:
        raise ValueError(f"Couldn't find optimizer with name '{name}'")


def wrap_optim_cls(cls: Type[Optimizer]) -> Type[Optimizer]:
    """Return PyTorch optimizer with additional context manager."""

    class ContextManagerOptim(cls):  # type:ignore
        # pylint:disable=missing-class-docstring,too-few-public-methods
        @contextlib.contextmanager
        def optimize(self):
            """Zero grads before yielding and step the optimizer upon exit."""
            self.zero_grad()
            yield
            self.step()

    return ContextManagerOptim


def link_optimizer(cls: Type[Optimizer], module: nn.Module, config: dict) -> Optimizer:
    """Construct optimizer tied to a module.

    Args:
        cls: the type of the optimizer
        module: the neural network module
        config: options to pass to the optimizer's constructor

    Returns:
        The optimizer instance
    """
    if issubclass(cls, (EKFAC, KFAC)):
        return cls(module, **config)
    return cls(module.parameters(), **config)
