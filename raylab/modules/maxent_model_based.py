"""Model-based Actor-Critic for the maximum entropy framework."""
from .model_actor_critic import AbstractModelActorCritic
from .stochastic_model_mixin import StochasticModelMixin
from .stochastic_actor_mixin import StochasticActorMixin, MaximumEntropyMixin
from .state_value_mixin import StateValueMixin


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
