"""Registry of modules for PyTorch policies."""
import torch

from .naf_module import NAFModule
from .ddpg_module import DDPGModule
from .sac_module import SACModule
from .simple_model_based import SimpleModelBased
from .svg_module import SVGModule
from .svg_realnvp_actor import SVGRealNVPActor
from .mapo_module import MAPOModule
from .maxent_model_based import MaxEntModelBased
from .on_policy_actor_critic import OnPolicyActorCritic
from .trpo_flows import TRPOFlows
from .trpo_tang2018 import TRPOTang2018

MODULES = {
    "NAFModule": NAFModule,
    "DDPGModule": DDPGModule,
    "SACModule": SACModule,
    "SimpleModelBased": SimpleModelBased,
    "SVGModule": SVGModule,
    "MAPOModule": MAPOModule,
    "MaxEntModelBased": MaxEntModelBased,
    "OnPolicyActorCritic": OnPolicyActorCritic,
    "TRPOFlows": TRPOFlows,
    "TRPOTang2018": TRPOTang2018,
    "SVGRealNVPActor": SVGRealNVPActor,
}


def get_module(obs_space, action_space, config):
    """Retrieve and construct module of given name."""
    type_ = config.pop("type")
    torch_script = config.get("torch_script")
    module = MODULES[type_](obs_space, action_space, config)
    return torch.jit.script(module) if torch_script else module
