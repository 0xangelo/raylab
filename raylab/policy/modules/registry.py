"""Registry of modules for PyTorch policies."""
import torch.nn as nn
from gym.spaces import Space

from .ddpg import DDPG
from .mb_ddpg import ModelBasedDDPG
from .mb_sac import ModelBasedSAC
from .naf import NAF
from .sac import SAC
from .v0.catalog import get_module as get_module_v0

MODULES = {}


class RepeatedModuleNameError(Exception):
    """Exception raised for attempting to register a repeated module name.

    Args:
        cls: NN module class
    """

    def __init__(self, cls: type):
        super().__init__(f"Module class {cls.__name__} already in catalog")


def register(cls: type):
    """Register module class in catalog by class name.

    Adds module to global registry dict

    Args:
        cls: NN module class

    Raises:
        RepeatedModuleNameError: If `cls` is already in registry
    """
    try:
        MODULES[cls.__name__] = cls
    except KeyError:
        raise RepeatedModuleNameError(cls)

    return cls


def get_module(obs_space: Space, action_space: Space, config: dict) -> nn.Module:
    """Retrieve and construct module of given name.

    Args:
        obs_space: Observation space
        action_space: Action space
        config: Configurations for module construction and initialization
    """
    type_ = config.pop("type")
    if type_ not in MODULES:
        return get_module_v0(obs_space, action_space, {"type": type_, **config})

    cls = MODULES[type_]
    spec = cls.spec_cls.from_dict(config)
    return cls(obs_space, action_space, spec)


for _cls in (DDPG, NAF, SAC, ModelBasedDDPG, ModelBasedSAC):
    register(_cls)
