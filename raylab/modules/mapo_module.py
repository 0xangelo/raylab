"""MAPO Architecture with disjoint model, actor, and critic."""
from ray.rllib.utils import deep_update

from .model_actor_critic import AbstractModelActorCritic
from .stochastic_model_mixin import StochasticModelMixin
from .deterministic_actor_mixin import DeterministicActorMixin
from .action_value_mixin import ActionValueMixin


BASE_CONFIG = {
    "torch_script": False,
    "actor": {
        "beta": 1.2,
        "smooth_target_policy": False,
        "target_gaussian_sigma": 0.3,
        "perturbed_policy": False,
        "encoder": {
            "units": (32, 32),
            "activation": "ReLU",
            "initializer_options": {"name": "xavier_uniform"},
            "layer_norm": False,
        },
    },
    "critic": {
        "double_q": False,
        "encoder": {
            "units": (32, 32),
            "activation": "ReLU",
            "initializer_options": {"name": "xavier_uniform"},
            "delay_action": True,
        },
    },
    "model": {
        "residual": False,
        "input_dependent_scale": False,
        "encoder": {
            "units": (32, 32),
            "activation": "ReLU",
            "initializer_options": {"name": "xavier_uniform"},
            "delay_action": True,
        },
    },
}


class MAPOModule(
    StochasticModelMixin,
    DeterministicActorMixin,
    ActionValueMixin,
    AbstractModelActorCritic,
):
    """Module architecture used in Model-Aware Policy Optimization."""

    # pylint:disable=abstract-method

    def __init__(self, obs_space, action_space, config):
        config = deep_update(BASE_CONFIG, config, False, ["actor", "critic", "model"])
        super().__init__(obs_space, action_space, config)
