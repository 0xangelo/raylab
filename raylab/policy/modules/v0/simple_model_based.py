"""Model-based architecture with disjoint model, actor, and critic."""
from .abstract import AbstractModelActorCritic
from .mixins import StateValueMixin
from .mixins import StochasticActorMixin
from .mixins import StochasticModelMixin


# pylint:disable=abstract-method
class SimpleModelBased(
    StochasticModelMixin,
    StochasticActorMixin,
    StateValueMixin,
    AbstractModelActorCritic,
):
    """Module architecture with stochastic actor and model, and state value function."""
