"""PyTorch related utilities."""
import numpy as np
import torch


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
