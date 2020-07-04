"""Utilities for module initialization."""
import functools
import inspect
from typing import Callable
from typing import Optional
from typing import Union

import torch.nn as nn
from torch import Tensor


def get_initializer(name: Optional[str]) -> Callable[[Tensor], None]:
    """Return initializer function given its name.

    Arguments:
        name: The initializer function's name. If None, returns a no-op callable
    """
    if name is None:
        return lambda _: None

    name_ = name + "_"
    if name in dir(nn.init) and name_ in dir(nn.init):
        func = getattr(nn.init, name_)
        return func
    raise ValueError(f"Couldn't find initializer with name '{name}'")


NONLINEARITY_MAP = {
    "Sigmoid": "sigmoid",
    "Tanh": "tanh",
    "ReLU": "relu",
    "ELU": "relu",
    "LeakyReLU": "leaky_relu",
}


def initialize_(
    name: Optional[str] = None, activation: Union[str, dict] = None, **options
) -> Callable[[nn.Module], None]:
    """Return a callable to apply an initializer with the given name and options.

    If `gain` is part of the initializer's argspec and is not specified in options,
    the recommended value from `torch.nn.init.calculate_gain` is used.

    Arguments:
        name: Initializer function name
        activation: Optional specification of the activation function that
            follows linear layers
        **options: Keyword arguments to pass to the initializer

    Returns:
        A callable to be used with `nn.Module.apply`.
    """

    initializer = get_initializer(name)

    if isinstance(activation, dict):
        activation = activation["name"]
        options.update(activation.get("options", {}))

    if (
        activation in NONLINEARITY_MAP
        and "gain" not in options
        and "gain" in inspect.signature(initializer).parameters
    ):
        recommended_gain = nn.init.calculate_gain(
            NONLINEARITY_MAP[activation], param=options.get("negative_slope")
        )
        options["gain"] = recommended_gain
    func_ = functools.partial(initializer, **options)

    def init(module):
        if isinstance(module, nn.Linear):
            func_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    return init
