"""Off-policy Actor-Critic with Normalizing Flows density approximation."""
from raylab.utils.dictionaries import deep_merge

from .abstract import AbstractActorCritic
from .mixins import ActionValueMixin
from .mixins import MaximumEntropyMixin
from .mixins import NormalizingFlowActorMixin


BASE_CONFIG = {"actor": {}, "critic": {}, "entropy": {}}


class OffPolicyNFAC(
    NormalizingFlowActorMixin,
    ActionValueMixin,
    MaximumEntropyMixin,
    AbstractActorCritic,
):
    """Actor-Critic module with stochastic actor and action-value critics."""

    # pylint:disable=abstract-method

    def __init__(self, obs_space, action_space, config):
        config = deep_merge(BASE_CONFIG, config, False, ["actor", "critic", "entropy"])
        super().__init__(obs_space, action_space, config)
