"""Model-based architecture with Normalizing Flows."""
from .abstract import AbstractModelActorCritic
from .mixins import (
    MaximumEntropyMixin,
    NormalizingFlowModelMixin,
    NormalizingFlowActorMixin,
    StateValueMixin,
)


# pylint:disable=abstract-method,too-many-ancestors
class NFMBRL(
    MaximumEntropyMixin,
    NormalizingFlowModelMixin,
    NormalizingFlowActorMixin,
    StateValueMixin,
    AbstractModelActorCritic,
):
    """
    Module architecture with normalizing flow actor and model, and state value function.
    """
