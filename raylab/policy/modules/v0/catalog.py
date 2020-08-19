"""Registry of old modules for PyTorch policies."""
import torch.nn as nn
from gym.spaces import Space

from .maxent_model_based import MaxEntModelBased
from .on_policy_actor_critic import OnPolicyActorCritic
from .simple_model_based import SimpleModelBased
from .svg_module import SVGModule

MODULES = {
    cls.__name__ + "-v0": cls
    for cls in (SimpleModelBased, SVGModule, MaxEntModelBased, OnPolicyActorCritic)
}


def get_module(obs_space: Space, action_space: Space, config: dict) -> nn.Module:
    """Retrieve and construct module of given name.

    Args:
        obs_space: Observation space
        action_space: Action space
        config: Configurations for module construction and initialization
    """
    type_ = config.pop("type")
    return MODULES[type_](obs_space, action_space, config)
