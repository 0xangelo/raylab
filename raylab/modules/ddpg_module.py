"""Actor-Critic architecture popularized by DDPG."""
from ray.rllib.utils import merge_dicts

from .actor_critic import AbstractActorCritic
from .deterministic_actor_mixin import DeterministicActorMixin
from .action_value_mixin import ActionValueMixin


BASE_CONFIG = {
    "torch_script": False,
    "double_q": False,
    "exploration": None,
    "exploration_gaussian_sigma": 0.3,
    "smooth_target_policy": False,
    "target_gaussian_sigma": 0.3,
    "actor": {
        "units": (32, 32),
        "activation": "ReLU",
        "initializer_options": {"name": "xavier_uniform"},
        "beta": 1.2,
    },
    "critic": {
        "units": (32, 32),
        "activation": "ReLU",
        "initializer_options": {"name": "xavier_uniform"},
        "delay_action": True,
    },
}


class DDPGModule(DeterministicActorMixin, ActionValueMixin, AbstractActorCritic):
    """Actor-Critic module with deterministic actor and action-value critics."""

    # pylint:disable=abstract-method

    def __init__(self, obs_space, action_space, config):
        super().__init__(obs_space, action_space, merge_dicts(BASE_CONFIG, config))
