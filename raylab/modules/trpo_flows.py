"""Trust-Region Policy Optimization with RealNVP density approximation."""
from ray.rllib.utils import deep_update

from .abstract import AbstractActorCritic
from .mixins import StateValueMixin, NormalizingFlowActorMixin


BASE_CONFIG = {"torch_script": True, "critic": {}}


class TRPOFlows(NormalizingFlowActorMixin, StateValueMixin, AbstractActorCritic):
    """Actor-Critic module with stochastic actor and state-value critics."""

    # pylint:disable=abstract-method

    def __init__(self, obs_space, action_space, config):
        config = deep_update(BASE_CONFIG, config, False, ["actor", "critic"])
        config["critic"]["target_vf"] = False
        super().__init__(obs_space, action_space, config)
