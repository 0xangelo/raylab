"""Modularized losses/procedures for RL algorithms."""
from .abstract import Loss
from .dyna import DynaQLearning
from .isfv_iteration import ISFittedVIteration, ISSoftVIteration
from .mage import MAGE
from .maximum_entropy import MaximumEntropyDual
from .mle import MaximumLikelihood
from .policy_gradient import (
    ActionDPG,
    DeterministicPolicyGradient,
    ReparameterizedSoftPG,
)
from .q_learning import FittedQLearning
from .svg import OneStepSoftSVG, OneStepSVG, TrajectorySVG

__all__ = [
    "Loss",
    "DynaQLearning",
    "FittedQLearning",
    "ActionDPG",
    "DeterministicPolicyGradient",
    "ISFittedVIteration",
    "ISSoftVIteration",
    "MaximumEntropyDual",
    "MAGE",
    "MaximumLikelihood",
    "OneStepSVG",
    "OneStepSoftSVG",
    "ReparameterizedSoftPG",
    "TrajectorySVG",
]
