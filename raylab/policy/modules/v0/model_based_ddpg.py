"""NN Architecture with disjoint model, actor, and critic."""
from raylab.utils.dictionaries import deep_merge

from .abstract import AbstractModelActorCritic
from .mixins import ActionValueMixin
from .mixins import DeterministicActorMixin
from .mixins import StochasticModelMixin


BASE_CONFIG = {
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


class ModelBasedDDPG(
    StochasticModelMixin,
    DeterministicActorMixin,
    ActionValueMixin,
    AbstractModelActorCritic,
):
    """Module architecture used in MAGE."""

    # pylint:disable=abstract-method

    def __init__(self, obs_space, action_space, config):
        config = deep_merge(BASE_CONFIG, config, False, ["actor", "critic", "model"])
        super().__init__(obs_space, action_space, config)
