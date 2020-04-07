"""Actor-Critic architecture used in Trust-Region Policy Optimization (TRPO)."""
from ray.rllib.utils import merge_dicts

from .actor_critic import AbstractActorCritic
from .stochastic_actor_mixin import StochasticActorMixin
from .state_value_mixin import StateValueMixin


BASE_CONFIG = {
    "torch_script": True,
    "actor": {
        "units": (32, 32),
        "activation": "Tanh",
        "initializer_options": {"name": "xavier_uniform"},
        "input_dependent_scale": False,
    },
    "critic": {
        "units": (32, 32),
        "activation": "Tanh",
        "initializer_options": {"name": "xavier_uniform"},
        "target_vf": False,
    },
}


class TRPOModule(
    StochasticActorMixin, StateValueMixin, AbstractActorCritic,
):
    """Actor-Critic module with stochastic actor and state-value critics."""

    # pylint:disable=abstract-method

    def __init__(self, obs_space, action_space, config):
        super().__init__(obs_space, action_space, merge_dicts(BASE_CONFIG, config))
