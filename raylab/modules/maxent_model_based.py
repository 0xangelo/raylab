"""Model-based Actor-Critic for the maximum entropy framework."""
from .abstract import AbstractModelActorCritic
from .mixins import (
    StochasticModelMixin,
    StochasticActorMixin,
    MaximumEntropyMixin,
    StateValueMixin,
)


# pylint:disable=abstract-method,too-many-ancestors
class MaxEntModelBased(
    StochasticModelMixin,
    StochasticActorMixin,
    StateValueMixin,
    MaximumEntropyMixin,
    AbstractModelActorCritic,
):
    """
    Module architecture with stochastic actor and model, state-value function, and
    entropy coefficient.
    """
