"""Registry of modules for PyTorch policies."""
import torch

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


def get_module(obs_space, action_space, config):
    """Retrieve and construct module of given name."""
    name = config.pop("name")
    torch_script = config.get("torch_script")
    module = MODULES[name](obs_space, action_space, config)
    return torch.jit.script(module) if torch_script else module
