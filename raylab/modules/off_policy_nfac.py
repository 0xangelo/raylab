"""Off-policy Actor-Critic with Normalizing Flows density approximation."""
from ray.rllib.utils import deep_update

from .abstract import AbstractActorCritic
from .mixins import ActionValueMixin, NormalizingFlowActorMixin, MaximumEntropyMixin


BASE_CONFIG = {"torch_script": True, "actor": {}, "critic": {}}


class OffPolicyNFAC(
    NormalizingFlowActorMixin,
    ActionValueMixin,
    MaximumEntropyMixin,
    AbstractActorCritic,
):
    """Actor-Critic module with stochastic actor and action-value critics."""

    # pylint:disable=abstract-method

    def __init__(self, obs_space, action_space, config):
        config = deep_update(BASE_CONFIG, config, False, ["actor", "critic"])
        super().__init__(obs_space, action_space, config)
