"""Off-policy Actor-Critic with Normalizing Flows density approximation."""
from raylab.utils.dictionaries import deep_merge

from .abstract import AbstractActorCritic
from .mixins import ActionValueMixin, NormalizingFlowActorMixin, MaximumEntropyMixin


BASE_CONFIG = {"torch_script": True, "actor": {}, "critic": {}, "entropy": {}}


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
