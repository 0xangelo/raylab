"""Registry of modules for PyTorch policies."""
from .naf_module import NormalizedAdvantageFunction
from .deterministic_actor_critic import DeterministicActorCritic
from .stochastic_actor_critic import StochasticActorCritic, MaxEntActorCritic
from .model_actor_critic import ModelActorCritic
from .svg_module import SVGModelActorCritic

MODULES = {
    "NormalizedAdvantageFunction": NormalizedAdvantageFunction,
    "DeterministicActorCritic": DeterministicActorCritic,
    "StochasticActorCritic": StochasticActorCritic,
    "MaxEntActorCritic": MaxEntActorCritic,
    "ModelActorCritic": ModelActorCritic,
    "SVGModelActorCritic": SVGModelActorCritic,
}


def get_module(name, obs_space, action_space, config):
    """Retrieve and construct module of given name."""
    return MODULES[name](obs_space, action_space, config)
