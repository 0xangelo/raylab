"""PyTorch related utilities."""
import numpy as np
import torch
from torch.autograd import grad


def flat_grad(outputs, inputs, *args, **kwargs):
    """Compute gradients and return a flattened array."""
    params = list(inputs)
    grads = grad(outputs, params, *args, **kwargs)
    zeros = torch.zeros
    return torch.cat(
        [zeros(p.numel()) if g is None else g.flatten() for p, g in zip(params, grads)]
    )


def convert_to_tensor(arr, device):
    """Convert array-like object to tensor and cast it to appropriate device.

    Arguments:
        arr (object): array-like object which can be converted using `np.asarray`
        device (torch.device): device to cast the resulting tensor to

    Returns:
        The array converted to a `torch.Tensor`.
    """
    if torch.is_tensor(arr):
        return arr.to(device)
    tensor = torch.from_numpy(np.asarray(arr))
    if tensor.dtype == torch.double:
        tensor = tensor.float()
    return tensor.to(device)
