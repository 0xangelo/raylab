"""SVG Module with RealNVP density approximation for the policy."""

from .abstract import AbstractModelActorCritic
from .mixins import NormalizingFlowActorMixin
from .mixins import StateValueMixin
from .mixins import SVGModelMixin


# pylint:disable=abstract-method
class SVGRealNVPActor(
    SVGModelMixin, NormalizingFlowActorMixin, StateValueMixin, AbstractModelActorCritic
):
    """SVG Module with RealNVP density approximation for the policy."""
