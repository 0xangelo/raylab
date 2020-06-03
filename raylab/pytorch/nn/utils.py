# pylint:disable=missing-docstring
# pylint:enable=missing-docstring
import functools

import torch
import torch.nn as nn


def get_activation(activation):
    """Return activation module type from string.

    Arguments:
        activation (str, dict or None): the activation function's description
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
    raise ValueError(f"Couldn't find activation with name '{name}'")


def update_polyak(from_module, to_module, polyak):
    """Update parameters between modules by polyak averaging.

    Arguments:
        from_module (nn.Module): Module whose parameters are targets.
        to_module (nn.Module): Module whose parameters are updated towards the targets.
        polyak (float): Averaging factor. The higher it is, the slower the parameters
            are updated.
    """
    for source, target in zip(from_module.parameters(), to_module.parameters()):
        target.data.mul_(polyak).add_(source.data, alpha=1 - polyak)


def perturb_module_params(module, target_module, stddev):
    """Load state dict from another module and perturb parameters not in layer norms.

    Arguments:
        module (nn.Module): the module to perturb
        target_module (nn.Module): the module to copy from
        stddev (float): the gaussian standard deviation with which to perturb parameters
            excluding those from layer norms
    """
    module.load_state_dict(target_module.state_dict())

    layer_norms = (m for m in module.modules() if isinstance(m, nn.LayerNorm))
    layer_norm_params = set(p for m in layer_norms for p in m.parameters())
    to_perturb = (p for p in module.parameters() if p not in layer_norm_params)

    for param in to_perturb:
        param.data.add_(torch.randn_like(param) * stddev)


def _sum_rightmost(value, dim: int):
    r"""
    Sum out ``dim`` many rightmost dimensions of a given tensor.

    Args:
        value (Tensor): A tensor of ``.dim()`` at least ``dim``.
        dim (int): The number of rightmost dims to sum out.
    """
    if dim == 0:
        return value
    required_shape = value.shape[:-dim] + (-1,)
    return value.reshape(required_shape).sum(-1)


def _multiply_rightmost(value, dim: int):
    r"""
    Multiply out ``dim`` many rightmost dimensions of a given tensor.

    Args:
        value (Tensor): A tensor of ``.dim()`` at least ``dim``.
        dim (int): The number of rightmost dims to multiply out.
    """
    if dim == 0:
        return value
    required_shape = value.shape[:-dim] + (-1,)
    return value.reshape(required_shape).prod(-1)
