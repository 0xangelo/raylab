# pylint:disable=missing-module-docstring
import functools
from typing import Callable
from typing import Union

import torch.nn as nn

from . import activation as activations


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

    if name in dir(activations):
        cls = getattr(activations, name)
        if issubclass(cls, nn.Module):
            return functools.partial(cls, **options)

    raise ValueError(f"Couldn't find activation with name '{name}'")
