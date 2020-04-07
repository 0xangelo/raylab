"""Actor-Critic architecture used in Soft Actor-Critic (SAC)."""
from ray.rllib.utils import merge_dicts

from .actor_critic import AbstractActorCritic
from .stochastic_actor_mixin import MaximumEntropyMixin, StochasticActorMixin
from .action_value_mixin import ActionValueMixin


BASE_CONFIG = {
    "double_q": True,
    "torch_script": False,
    "actor": {
        "units": (32, 32),
        "activation": "ReLU",
        "initializer_options": {"name": "xavier_uniform"},
        "input_dependent_scale": True,
    },
    "critic": {
        "units": (32, 32),
        "activation": "ReLU",
        "initializer_options": {"name": "xavier_uniform"},
        "delay_action": True,
    },
}


class SACModule(
    StochasticActorMixin, ActionValueMixin, MaximumEntropyMixin, AbstractActorCritic
):
    """Actor-Critic module with stochastic actor and action-value critics."""

    # pylint:disable=abstract-method

    def __init__(self, obs_space, action_space, config):
        super().__init__(obs_space, action_space, merge_dicts(BASE_CONFIG, config))
