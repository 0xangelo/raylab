"""Registry of modules for PyTorch policies."""
from .naf_module import NAFModule
from .deterministic_actor_critic import DeterministicActorCritic
from .stochastic_actor_critic import StochasticActorCritic
from .model_actor_critic import ModelActorCritic
from .svg_module import SVGModelActorCritic

MODULES = {
    "NAFModule": NAFModule,
    "DeterministicActorCritic": DeterministicActorCritic,
    "StochasticActorCritic": StochasticActorCritic,
    "ModelActorCritic": ModelActorCritic,
    "SVGModelActorCritic": SVGModelActorCritic,
}


def get_module(name, obs_space, action_space, config):
    """Retrieve and construct module of given name."""
    return MODULES[name](obs_space, action_space, config)
