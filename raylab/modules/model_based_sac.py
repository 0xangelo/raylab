"""Model-based Soft Actor-Critic architecture."""

from .abstract import AbstractModelActorCritic
from .mixins import ActionValueMixin
from .mixins import MaximumEntropyMixin
from .mixins import StochasticActorMixin
from .mixins import StochasticModelMixin


# pylint:disable=abstract-method,too-many-ancestors
class ModelBasedSAC(
    StochasticModelMixin,
    StochasticActorMixin,
    ActionValueMixin,
    MaximumEntropyMixin,
    AbstractModelActorCritic,
):
    """
    Module architecture with stochastic actor and model, action-value function, and
    entropy coefficient.
    """
