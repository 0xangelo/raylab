"""Utilities for manipulating neural network modules."""
import functools
from typing import Callable
from typing import Union

import torch
import torch.nn as nn

from . import modules


def get_activation(activation: Union[str, dict, None]) -> Callable[[], nn.Module]:
    """Return activation module constructor from specification.

    Args:
        activation: the activation function's specification. Can be either
            a string with the activation module class' name, a dict with the
            name as the `name` field and additional keyword arguments, or None,
            in which case the indentity module class is returned.

    Raises:
        ValueError: If the type corresponding to the activation's name cannot
            be found.
    """
    if activation is None:
        return nn.Identity

    if isinstance(activation, dict):
        name = activation["name"]
        options = activation.get("options", {})
    else:
        name = activation
        options = {}

    if name in dir(nn.modules.activation):
        cls = getattr(nn.modules.activation, name)
        if issubclass(cls, nn.Module):
            return functools.partial(cls, **options)

    if name in dir(modules.activation):
        cls = getattr(modules.activation, name)
        if issubclass(cls, nn.Module):
            return functools.partial(cls, **options)

    raise ValueError(f"Couldn't find activation with name '{name}'")


def update_polyak(from_module: nn.Module, to_module: nn.Module, polyak: float):
    """Update parameters between modules by polyak averaging.

    Args:
        from_module: Module whose parameters are targets.
        to_module: Module whose parameters are updated towards the targets.
        polyak: Averaging factor. The higher it is, the slower the parameters
            are updated.
    """
    for source, target in zip(from_module.parameters(), to_module.parameters()):
        target.data.mul_(polyak).add_(source.data, alpha=1 - polyak)


def perturb_params(target: nn.Module, origin: nn.Module, stddev: float):
    """Set the parameters of a module to a noisy version of another's.

    Loads state dict from the origin module and perturbs the parameters of the
    target module. Layer normalization parameters are ignored.

    Args:
        target: the module to perturb
        origin: the module to copy from
        stddev: the gaussian standard deviation of the noise added to the origin
            model's parameters
    """
    target.load_state_dict(origin.state_dict())

    layer_norms = (m for m in target.modules() if isinstance(m, nn.LayerNorm))
    layer_norm_params = set(p for m in layer_norms for p in m.parameters())
    to_perturb = (p for p in target.parameters() if p not in layer_norm_params)

    for param in to_perturb:
        param.data.add_(torch.randn_like(param) * stddev)
