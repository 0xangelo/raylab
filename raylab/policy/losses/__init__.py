"""Modularized losses/procedures for RL algorithms."""
from .isfv_iteration import ISFittedVIteration
from .isfv_iteration import ISSoftVIteration
from .mage import MAGE
from .maximum_entropy import MaximumEntropyDual
from .mle import MaximumLikelihood
from .policy_gradient import ActionDPG
from .policy_gradient import DeterministicPolicyGradient
from .policy_gradient import ReparameterizedSoftPG
from .q_learning import FittedQLearning
from .svg import OneStepSoftSVG
from .svg import OneStepSVG
from .svg import TrajectorySVG


__all__ = [
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
