"""Model-based architecture with Normalizing Flows."""

from .abstract import AbstractModelActorCritic
from .mixins import MaximumEntropyMixin
from .mixins import NormalizingFlowActorMixin
from .mixins import NormalizingFlowModelMixin
from .mixins import StateValueMixin


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
