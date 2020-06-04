"""Utilities for module initialization."""
import functools
import inspect

import torch.nn as nn


def get_initializer(name):
    """Return initializer function given its name.

    Arguments:
        name (str): the initializer function's name
    """
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


def initialize_(name, activation=None, **options):
    """Return a callable to apply an initializer with the given name and options.

    If `gain` is part of the initializer's argspec and is not specified in options,
    the recommended value from `nn.init.calculate_gain` is used.

    Arguments:
        name (str): name of initializer function
        activation (str, dict): activation function following linear layer, optional
        **options: keyword arguments to be passed to the initializer

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
