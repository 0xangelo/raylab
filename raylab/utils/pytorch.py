"""PyTorch related utilities."""
import functools

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
        name (str): string representing the name of the optimizer

    Returns:
        The corresponding `torch.optim.Optimizer` subclass
    """
    if name in dir(torch.optim):
        cls = getattr(torch.optim, name)
        if issubclass(cls, torch.optim.Optimizer) and cls is not torch.optim.Optimizer:
            return cls
    raise ValueError(f"Couldn't find optimizer with name '{name}'")


def get_activation(name):
    """Return activation module type from string.

    Arguments:
        name (str): string representing the name of activation function
    """
    if name in dir(nn.modules.activation):
        cls = getattr(nn.modules.activation, name)
        if issubclass(cls, nn.Module):
            return cls
    raise ValueError(f"Couldn't find activation with name '{name}'")


def get_initializer(name):
    """Return initializer function given its name.

    Arguments:
        name (str): string representing the name of initializer function
    """
    name_ = name + "_"
    if name in dir(nn.init) and name_ in dir(nn.init):
        func = getattr(nn.init, name_)
        return func
    raise ValueError(f"Couldn't find initializer with name '{name}'")


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


def initialize_(name, **options):
    """Return a callable to apply an initializer with the given name and options.

    Arguments:
        name (str): name of initializer function
        **options: keyword arguments to be passed to the initializer

    Returns:
        A callable to be used with `nn.Module.apply`.
    """

    initializer = get_initializer(name)
    func_ = functools.partial(initializer, **options)

    def init(module):
        if isinstance(module, nn.Linear):
            func_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    return init


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


def trace(func):
    """
    Wrapps and automatically traces an instance function on first call.

    Arguments:
        func (callable): the callable to be converted to TorchScript. Should not
            have any input-dependent control flow.
    """
    method_name = "_traced_" + func.__name__

    @functools.wraps(func)
    def wrapped(self, *args):
        if not hasattr(self, method_name):
            traced = torch.jit.trace(functools.partial(func, self), args)
            setattr(self, method_name, traced)
        return getattr(self, method_name)(*args)

    return wrapped
