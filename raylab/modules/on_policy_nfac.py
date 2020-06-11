"""On-policy Actor-Critic with Normalizing Flows density approximation."""
from raylab.utils.dictionaries import deep_merge

from .abstract import AbstractActorCritic
from .mixins import NormalizingFlowActorMixin
from .mixins import StateValueMixin


BASE_CONFIG = {"actor": {}, "critic": {}}


class OnPolicyNFAC(NormalizingFlowActorMixin, StateValueMixin, AbstractActorCritic):
    """Actor-Critic module with stochastic actor and state-value critics."""

    # pylint:disable=abstract-method

    def __init__(self, obs_space, action_space, config):
        config = deep_merge(BASE_CONFIG, config, False, ["actor", "critic"])
        config["critic"]["target_vf"] = False
        super().__init__(obs_space, action_space, config)
