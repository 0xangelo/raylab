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
        cls = getattr(torch.optim, name)
        if issubclass(cls, torch.optim.Optimizer) and cls is not torch.optim.Optimizer:
            return cls
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


def get_activation(activation):
    """Return activation module type from string.

    Arguments:
        activation (str): name of activation function
    """
    if not isinstance(activation, str):
        raise ValueError(
            "'activation' must be a string type, got '{}'".format(type(activation))
        )

    if activation in dir(torch.nn.modules.activation):
        cls = getattr(torch.nn.modules.activation, activation)
        if issubclass(cls, nn.Module):
            return cls
    raise ValueError("Unsupported activation name '{}'".format(activation))
