"""Model-based Actor-Critic for the maximum entropy framework."""
from .abstract import AbstractModelActorCritic
from .mixins import MaximumEntropyMixin
from .mixins import StateValueMixin
from .mixins import StochasticActorMixin
from .mixins import StochasticModelMixin


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
