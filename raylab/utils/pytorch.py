"""PyTorch related utilities."""
import numpy as np
import torch
import torch.nn as nn


def convert_to_tensor(arr, device):
    """Convert array-like object to tensor and cast it to appropriate device.

    Arguments:
        arr (object): array-like object which can be converted using `np.asarray`
        device (torch.device): device to cast the resulting tensor to

    Returns:
        The array converted to a `torch.Tensor`.
    """
    tensor = torch.from_numpy(np.asarray(arr))
    if tensor.dtype == torch.double:
        tensor = tensor.float()
    return tensor.to(device)


def get_optimizer_class(name):
    """Return the optimizer class given its name.

    Arguments:
        name (str): string representing the name of optimizer

    Returns:
        The corresponding `torch.optim.Optimizer` subclass
    """
    if name in dir(torch.optim):
        obj = getattr(torch.optim, name)
        if issubclass(obj, torch.optim.Optimizer) and obj is not torch.optim.Optimizer:
            return obj
    raise ValueError("Unsupported optimizer name '{}'".format(name))


def update_polyak(from_module, to_module, polyak):
    """Update parameters between modules by polyak averaging.

    Arguments:
        from_module (nn.Module): Module whose parameters are targets.
        to_module (nn.Module): Module whose parameters are updated towards the targets.
        polyak (float): Averaging factor. The higher it is, the slower the parameters
            are updated.
    """
    for source, target in zip(from_module.parameters(), to_module.parameters()):
        target.data.mul_(polyak).add_(1 - polyak, source.data)


def initialize_orthogonal(gain=1.0):
    """Initialize the parameters of a module as orthogonal matrices.

    Arguments:
        gain (float): scaling factor for the orthogonal initializer

    Returns:
        A callable to be used with `nn.Module.apply`.
    """

    def initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    return initialize
