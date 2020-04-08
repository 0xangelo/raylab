"""Trust-Region Policy Optimization with RealNVP density approximation."""
from ray.rllib.utils import deep_update

from .actor_critic import AbstractActorCritic
from .state_value_mixin import StateValueMixin
from .realnvp_actor_mixin import RealNVPActorMixin


BASE_CONFIG = {
    "torch_script": True,
    "actor": {
        "obs_encoder": {
            "units": (64, 64),
            "activation": "ELU",
            "layer_norm": False,
            "initializer_options": {"name": "xavier_uniform"},
        },
        "num_flows": 4,
        "flow_mlp": {
            "units": (24,) * 4,
            "activation": "ELU",
            "layer_norm": False,
            "initializer_options": {"name": "xavier_uniform"},
        },
    },
    "critic": {
        "units": (64, 64),
        "activation": "ELU",
        "initializer_options": {"name": "xavier_uniform"},
    },
}


class TRPORealNVP(RealNVPActorMixin, StateValueMixin, AbstractActorCritic):
    """Actor-Critic module with stochastic actor and state-value critics."""

    # pylint:disable=abstract-method

    def __init__(self, obs_space, action_space, config):
        config = deep_update(BASE_CONFIG, config, False, ["actor", "critic"])
        config["critic"]["target_vf"] = False
        super().__init__(obs_space, action_space, config)
