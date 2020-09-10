"""Registry of old modules for PyTorch policies."""
import torch.nn as nn
from gym.spaces import Space

from .on_policy_actor_critic import OnPolicyActorCritic

MODULES = {cls.__name__ + "-v0": cls for cls in (OnPolicyActorCritic,)}


def get_module(obs_space: Space, action_space: Space, config: dict) -> nn.Module:
    """Retrieve and construct module of given name.

    Args:
        obs_space: Observation space
        action_space: Action space
        config: Configurations for module construction and initialization
    """
    type_ = config.pop("type")
    return MODULES[type_](obs_space, action_space, config)
