"""Registry of modules for PyTorch policies."""
import torch.nn as nn
from gym.spaces import Space

from .ddpg import DDPG
from .mbddpg import MBDDPG
from .mbsac import MBSAC
from .sac import SAC
from .v0.catalog import get_module as get_v0_module

MODULES = {cls.__name__: cls for cls in (DDPG, MBDDPG, MBSAC, SAC)}


def get_module(obs_space: Space, action_space: Space, config: dict) -> nn.Module:
    """Retrieve and construct module of given name.

    Args:
        obs_space: Observation space
        action_space: Action space
        config: Configurations for module construction and initialization
    """
    type_ = config.pop("type")
    if type_ not in MODULES:
        return get_v0_module(obs_space, action_space, {"type": type_, **config})

    cls = MODULES[type_]
    spec = cls.spec_cls.from_dict(config)
    return cls(obs_space, action_space, spec)
