"""Network and configurations for modules with stochastic policies."""
import warnings
from dataclasses import dataclass
from dataclasses import field
from typing import Union

import torch.nn as nn
from dataclasses_json import DataClassJsonMixin
from gym.spaces import Box
from gym.spaces import Discrete

from .policy.stochastic import Alpha
from .policy.stochastic import MLPContinuousPolicy
from .policy.stochastic import MLPDiscretePolicy
from .policy.stochastic import MLPStochasticPolicy

MLPSpec = MLPStochasticPolicy.spec_cls


@dataclass
class StochasticActorSpec(DataClassJsonMixin):
    """Specifications for stochastic policy.

    Args:
        encoder: Specifications for building the multilayer perceptron state
            processor
        input_dependent_scale: Whether to parameterize the Gaussian standard
            deviation as a function of the state
        initial_entropy_coeff: Optional initial value of the entropy bonus term.
            The actor creates an `alpha` attribute with this initial value.
    """

    encoder: MLPSpec = field(default_factory=MLPSpec)
    input_dependent_scale: bool = False
    initial_entropy_coeff: float = 0.0

    def __post_init__(self):
        cls_name = type(self).__name__
        ent_coeff = self.initial_entropy_coeff
        if ent_coeff < 0:
            warnings.warn(f"Entropy coefficient is negative in {cls_name}: {ent_coeff}")


class StochasticActor(nn.Module):
    """NN with stochastic policy.

    Args:
        obs_space: Observation space
        action_space: Action space
        spec: Specifications for stochastic policy

    Attributes:
        policy: Stochastic policy to be learned
        alpha: Entropy bonus coefficient
    """

    # pylint:disable=abstract-method
    spec_cls = StochasticActorSpec

    def __init__(
        self,
        obs_space: Box,
        action_space: Union[Box, Discrete],
        spec: StochasticActorSpec,
    ):
        super().__init__()

        if isinstance(action_space, Box):
            policy = MLPContinuousPolicy(
                obs_space, action_space, spec.encoder, spec.input_dependent_scale
            )
        elif isinstance(action_space, Discrete):
            policy = MLPDiscretePolicy(obs_space, action_space, spec.encoder)
        else:
            raise ValueError(f"Unsopported action space type {type(action_space)}")

        self.policy = policy
        self.alpha = Alpha(spec.initial_entropy_coeff)
