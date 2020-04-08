"""Model-based architecture with disjoint model, actor, and critic."""
from .model_actor_critic import AbstractModelActorCritic
from .stochastic_model_mixin import StochasticModelMixin
from .stochastic_actor_mixin import StochasticActorMixin
from .state_value_mixin import StateValueMixin


# pylint:disable=abstract-method
class SimpleModelBased(
    StochasticModelMixin,
    StochasticActorMixin,
    StateValueMixin,
    AbstractModelActorCritic,
):
    """Module architecture with stochastic actor and model, and state value function."""
