# pylint:disable=missing-module-docstring
import torch


def sum_rightmost(value: torch.Tensor, dim: int):
    """Sum out ``dim`` many rightmost dimensions of a given tensor.

    Args:
        value: A tensor of ``.dim()`` at least ``dim``.
        dim: The number of rightmost dims to sum out.
    """
    if dim == 0:
        return value
    required_shape = value.shape[:-dim] + (-1,)
    return value.reshape(required_shape).sum(-1)


def multiply_rightmost(value: torch.Tensor, dim: int):
    """Multiply out ``dim`` many rightmost dimensions of a given tensor.

    Args:
        value: A tensor of ``.dim()`` at least ``dim``.
        dim: The number of rightmost dims to multiply out.
    """
    if dim == 0:
        return value
    required_shape = value.shape[:-dim] + (-1,)
    return value.reshape(required_shape).prod(-1)
