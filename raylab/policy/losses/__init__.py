"""Modularized losses/procedures for RL algorithms."""

from .cdq_learning import ClippedDoubleQLearning
from .cdq_learning import DynaSoftCDQLearning
from .cdq_learning import SoftCDQLearning
from .isfv_iteration import ISFittedVIteration
from .isfv_iteration import ISSoftVIteration
from .mage import MAGE
from .mapo import DAPO
from .mapo import MAPO
from .maximum_entropy import MaximumEntropyDual
from .mle import MaximumLikelihood
from .mle import ModelEnsembleMLE
from .paml import SPAML
from .policy_gradient import ActionDPG
from .policy_gradient import DeterministicPolicyGradient
from .policy_gradient import ReparameterizedSoftPG
from .svg import OneStepSoftSVG
from .svg import OneStepSVG
from .svg import TrajectorySVG


__all__ = [
    "ClippedDoubleQLearning",
    "ActionDPG",
    "DeterministicPolicyGradient",
    "ISFittedVIteration",
    "ISSoftVIteration",
    "MaximumEntropyDual",
    "MAGE",
    "DAPO",
    "MAPO",
    "MaximumLikelihood",
    "ModelEnsembleMLE",
    "PAML",
    "OneStepSVG",
    "OneStepSoftSVG",
    "ReparameterizedSoftPG",
    "SoftCDQLearning",
    "DynaSoftCDQLearning",
    "TrajectorySVG",
]
