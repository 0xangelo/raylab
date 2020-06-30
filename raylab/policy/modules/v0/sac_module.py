"""Actor-Critic architecture used in Soft Actor-Critic (SAC)."""
from raylab.utils.dictionaries import deep_merge

from .abstract import AbstractActorCritic
from .mixins import ActionValueMixin
from .mixins import MaximumEntropyMixin
from .mixins import StochasticActorMixin


BASE_CONFIG = {
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
    "entropy": {"initial_alpha": 0.05},
}


class SACModule(
    StochasticActorMixin, ActionValueMixin, MaximumEntropyMixin, AbstractActorCritic
):
    """Actor-Critic module with stochastic actor and action-value critics."""

    # pylint:disable=abstract-method

    def __init__(self, obs_space, action_space, config):
        config = deep_merge(BASE_CONFIG, config, False, ["actor", "critic", "entropy"])
        super().__init__(obs_space, action_space, config)
