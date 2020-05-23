"""Modularized losses/procedures for RL algorithms."""

from .cdq_learning import ClippedDoubleQLearning
from .cdq_learning import SoftCDQLearning
from .daml import DPGAwareModelLearning
from .isfv_iteration import ISFittedVIteration
from .isfv_iteration import ISSoftVIteration
from .mle import MaximumLikelihood
from .policy_gradient import DeterministicPolicyGradient
from .policy_gradient import ModelAwareDPG
from .policy_gradient import ReparameterizedSoftPG
from .svg import OneStepSVG
from .svg import TrajectorySVG


__all__ = [
    "ClippedDoubleQLearning",
    "DeterministicPolicyGradient",
    "DPGAwareModelLearning",
    "ISFittedVIteration",
    "ISSoftVIteration",
    "MaximumLikelihood",
    "ModelAwareDPG",
    "OneStepSVG",
    "ReparameterizedSoftPG",
    "SoftCDQLearning",
    "TrajectorySVG",
]
