"""Actor-Critic architecture used in Soft Actor-Critic (SAC)."""
from .actor_critic import AbstractActorCritic
from .stochastic_actor_mixin import MaximumEntropyMixin, StochasticActorMixin
from .action_value_mixin import ActionValueMixin


# pylint:disable=abstract-method
class SACModule(
    StochasticActorMixin, ActionValueMixin, MaximumEntropyMixin, AbstractActorCritic
):
    """Actor-Critic module with stochastic actor and action-value critics."""


# pylint:enable=abstract-method
