"""Registry of modules for PyTorch policies."""
import torch.nn as nn
from gym.spaces import Space

from .ddpg import DDPG
from .v0.ddpg_module import DDPGModule
from .v0.maxent_model_based import MaxEntModelBased
from .v0.model_based_ddpg import ModelBasedDDPG
from .v0.model_based_sac import ModelBasedSAC
from .v0.naf_module import NAFModule
from .v0.nfmbrl import NFMBRL
from .v0.off_policy_nfac import OffPolicyNFAC
from .v0.on_policy_actor_critic import OnPolicyActorCritic
from .v0.on_policy_nfac import OnPolicyNFAC
from .v0.sac_module import SACModule
from .v0.simple_model_based import SimpleModelBased
from .v0.svg_module import SVGModule
from .v0.svg_realnvp_actor import SVGRealNVPActor
from .v0.trpo_tang2018 import TRPOTang2018

MODULESv0 = {
    cls.__name__: cls
    for cls in (
        NAFModule,
        DDPGModule,
        SACModule,
        SimpleModelBased,
        SVGModule,
        MaxEntModelBased,
        ModelBasedDDPG,
        ModelBasedSAC,
        NFMBRL,
        OnPolicyActorCritic,
        OnPolicyNFAC,
        OffPolicyNFAC,
        TRPOTang2018,
        SVGRealNVPActor,
    )
}

MODULESv1 = {cls.__name__: cls for cls in (DDPG,)}


def get_module(obs_space: Space, action_space: Space, config: dict) -> nn.Module:
    """Retrieve and construct module of given name.

    Args:
        obs_space: Observation space
        action_space: Action space
        config: Configurations for module construction and initialization
    """
    type_ = config.pop("type")
    if type_ in MODULESv0:
        return MODULESv0[type_](obs_space, action_space, config)

    cls = MODULESv1[type_]
    spec = cls.spec_cls.from_dict(config)
    return cls(obs_space, action_space, spec)
