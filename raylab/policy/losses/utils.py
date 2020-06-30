"""Utilities for computing losses."""
import torch
import torch.nn as nn
from torch import Tensor


def clipped_action_value(
    obs: Tensor, actions: Tensor, critics: nn.ModuleList
) -> Tensor:
    """Compute the minimum of two action-value functions on state-action pairs."""
    value, _ = torch.cat([m(obs, actions) for m in critics], dim=-1).min(dim=-1)
    return value
