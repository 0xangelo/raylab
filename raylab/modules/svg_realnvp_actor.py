"""SVG Module with RealNVP density approximation for the policy."""
from .model_actor_critic import AbstractModelActorCritic
from .realnvp_actor_mixin import RealNVPActorMixin
from .svg_module import SVGModelMixin
from .state_value_mixin import StateValueMixin


# pylint:disable=abstract-method
class SVGRealNVPActor(
    SVGModelMixin, RealNVPActorMixin, StateValueMixin, AbstractModelActorCritic
):
    """SVG Module with RealNVP density approximation for the policy."""
