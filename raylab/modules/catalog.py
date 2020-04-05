"""Registry of modules for PyTorch policies."""
from .naf_module import NAFModule
from .ddpg_module import DDPGModule
from .sac_module import SACModule
from .svg_module import SVGModule
from .mapo_module import MAPOModule
from .trpo_module import TRPOModule
from .trpo_realnvp import TRPORealNVP
from .trpo_tang2018 import TRPOTang2018

MODULES = {
    "NAFModule": NAFModule,
    "DDPGModule": DDPGModule,
    "SACModule": SACModule,
    "SVGModule": SVGModule,
    "MAPOModule": MAPOModule,
    "TRPOModule": TRPOModule,
    "TRPORealNVP": TRPORealNVP,
    "TRPOTang2018": TRPOTang2018,
}


def get_module(name, obs_space, action_space, config):
    """Retrieve and construct module of given name."""
    return MODULES[name](obs_space, action_space, config)
