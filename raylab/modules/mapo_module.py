"""MAPO Architecture with disjoint model, actor, and critic."""
from .model_actor_critic import AbstractModelActorCritic
from .stochastic_model_mixin import StochasticModelMixin
from .deterministic_actor_mixin import DeterministicActorMixin
from .action_value_mixin import ActionValueMixin


# pylint:disable=abstract-method
class MAPOModule(
    StochasticModelMixin,
    DeterministicActorMixin,
    ActionValueMixin,
    AbstractModelActorCritic,
):
    """Module architecture used in Model-Aware Policy Optimization."""


# pylint:enable=abstract-method
