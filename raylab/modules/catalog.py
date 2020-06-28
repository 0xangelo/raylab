"""Registry of modules for PyTorch policies."""

from .ddpg import DDPG
from .ddpg_module import DDPGModule
from .maxent_model_based import MaxEntModelBased
from .model_based_ddpg import ModelBasedDDPG
from .model_based_sac import ModelBasedSAC
from .naf_module import NAFModule
from .nfmbrl import NFMBRL
from .off_policy_nfac import OffPolicyNFAC
from .on_policy_actor_critic import OnPolicyActorCritic
from .on_policy_nfac import OnPolicyNFAC
from .sac_module import SACModule
from .simple_model_based import SimpleModelBased
from .svg_module import SVGModule
from .svg_realnvp_actor import SVGRealNVPActor
from .trpo_tang2018 import TRPOTang2018

MODULES = {
    "NAFModule": NAFModule,
    "DDPGModule": DDPGModule,
    "SACModule": SACModule,
    "SimpleModelBased": SimpleModelBased,
    "SVGModule": SVGModule,
    "MaxEntModelBased": MaxEntModelBased,
    "ModelBasedDDPG": ModelBasedDDPG,
    "ModelBasedSAC": ModelBasedSAC,
    "NFMBRL": NFMBRL,
    "OnPolicyActorCritic": OnPolicyActorCritic,
    "OnPolicyNFAC": OnPolicyNFAC,
    "OffPolicyNFAC": OffPolicyNFAC,
    "TRPOTang2018": TRPOTang2018,
    "SVGRealNVPActor": SVGRealNVPActor,
}

MODULESv2 = {k.__name__: k for k in [DDPG]}


def get_module(obs_space, action_space, config):
    """Retrieve and construct module of given name."""
    type_ = config.pop("type")
    if type_ in MODULES:
        return MODULES[type_](obs_space, action_space, config)

    cls = MODULESv2[type_]
    spec = cls.spec_cls.from_dict(config)
    return cls(obs_space, action_space, spec)
