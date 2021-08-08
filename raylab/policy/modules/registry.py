"""Registry of modules for PyTorch policies."""
from gym.spaces import Space
from torch import nn

from .ddpg import DDPG
from .mage import MAGE
from .mb_ddpg import ModelBasedDDPG
from .mb_sac import ModelBasedSAC
from .naf import NAF
from .sac import SAC
from .sop import SOP
from .svg import SVG, SoftSVG
from .td3 import TD3
from .trpo import TRPO

MODULES = {}


class RepeatedModuleNameError(Exception):
    """Exception raised for attempting to register a repeated module name.

    Args:
        cls: NN module class
    """

    def __init__(self, cls: type):
        super().__init__(f"Module class {cls.__name__} already in catalog")


class UnknownModuleError(Exception):
    """Exception raised for attempting to query an unkown module.

    Args:
        name: NN module name
    """

    def __init__(self, name: str):
        super().__init__(f"No module registered with name '{name}' in catalog")


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
        raise RepeatedModuleNameError(cls)  # pylint:disable=raise-missing-from

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
        raise UnknownModuleError(type_)

    cls = MODULES[type_]
    spec = cls.spec_cls.from_dict(config)
    return cls(obs_space, action_space, spec)


for _cls in (
    DDPG,
    MAGE,
    NAF,
    SAC,
    SOP,
    ModelBasedDDPG,
    ModelBasedSAC,
    SVG,
    SoftSVG,
    TD3,
    TRPO,
):
    register(_cls)
