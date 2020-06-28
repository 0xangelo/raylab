"""Actor-Critic architecture used in most on-policy PG algorithms."""
from ray.rllib.utils import merge_dicts

from .abstract import AbstractActorCritic
from .mixins import StateValueMixin
from .mixins import StochasticActorMixin


BASE_CONFIG = {
    "actor": {
        "encoder": {"units": (32, 32), "activation": "Tanh"},
        "input_dependent_scale": False,
    },
    "critic": {
        "encoder": {"units": (32, 32), "activation": "Tanh"},
        "target_vf": False,
    },
}


class OnPolicyActorCritic(
    StochasticActorMixin, StateValueMixin, AbstractActorCritic,
):
    """Actor-Critic module with stochastic actor and state-value critics."""

    # pylint:disable=abstract-method

    def __init__(self, obs_space, action_space, config):
        super().__init__(obs_space, action_space, merge_dicts(BASE_CONFIG, config))
