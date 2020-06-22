"""Utilities for computing losses."""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from raylab.utils.annotations import RewardFn
from raylab.utils.annotations import TerminationFn


@dataclass
class EnvFunctions:
    """Collection of environment emulating functions."""

    reward: Optional[RewardFn] = None
    termination: Optional[TerminationFn] = None

    @property
    def initialized(self):
        """Whether or not all functions are set."""
        return self.reward is not None and self.termination is not None


def clipped_action_value(
    obs: Tensor, actions: Tensor, critics: nn.ModuleList
) -> Tensor:
    """Compute the minimum of two action-value functions on state-action pairs."""
    value, _ = torch.cat([m(obs, actions) for m in critics], dim=-1).min(dim=-1)
    return value
