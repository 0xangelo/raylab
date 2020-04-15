"""Actor-Critic architecture used in Soft Actor-Critic (SAC)."""
from ray.rllib.utils import deep_update

from .abstract import AbstractActorCritic
from .mixins import MaximumEntropyMixin, StochasticActorMixin, ActionValueMixin


BASE_CONFIG = {
    "torch_script": False,
    "actor": {
        "encoder": {
            "units": (400, 300),
            "activation": "ReLU",
            "initializer_options": {"name": "xavier_uniform"},
        },
        "input_dependent_scale": True,
    },
    "critic": {
        "double_q": False,
        "encoder": {
            "units": (400, 300),
            "activation": "ReLU",
            "initializer_options": {"name": "xavier_uniform"},
            "delay_action": True,
        },
    },
}


class SACModule(
    StochasticActorMixin, ActionValueMixin, MaximumEntropyMixin, AbstractActorCritic
):
    """Actor-Critic module with stochastic actor and action-value critics."""

    # pylint:disable=abstract-method

    def __init__(self, obs_space, action_space, config):
        config = deep_update(BASE_CONFIG, config, False, ["actor", "critic"])
        super().__init__(obs_space, action_space, config)
