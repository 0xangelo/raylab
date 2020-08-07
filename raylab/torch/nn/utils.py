"""Utilities for manipulating neural network modules."""
import torch
import torch.nn as nn

from .modules.utils import get_activation

__all__ = [
    "get_activation",
    "update_polyak",
    "perturb_params",
]


def update_polyak(from_module: nn.Module, to_module: nn.Module, polyak: float):
    """Update parameters between modules by polyak averaging.

    Args:
        from_module: Module whose parameters are targets.
        to_module: Module whose parameters are updated towards the targets.
        polyak: Averaging factor. The higher it is, the slower the parameters
            are updated.
    """
    for source, target in zip(from_module.parameters(), to_module.parameters()):
        target.data.mul_(polyak).add_(source.data, alpha=1 - polyak)


def perturb_params(target: nn.Module, origin: nn.Module, stddev: float):
    """Set the parameters of a module to a noisy version of another's.

    Loads state dict from the origin module and perturbs the parameters of the
    target module. Layer normalization parameters are ignored.

    Args:
        target: the module to perturb
        origin: the module to copy from
        stddev: the gaussian standard deviation of the noise added to the origin
            model's parameters
    """
    target.load_state_dict(origin.state_dict())

    layer_norms = (m for m in target.modules() if isinstance(m, nn.LayerNorm))
    layer_norm_params = set(p for m in layer_norms for p in m.parameters())
    to_perturb = (p for p in target.parameters() if p not in layer_norm_params)

    for param in to_perturb:
        param.data.add_(torch.randn_like(param) * stddev)
