"""Modularized losses/procedures for RL algorithms."""

from .cdq_learning import ClippedDoubleQLearning
from .cdq_learning import SoftCDQLearning
from .daml import DPGAwareModelLearning
from .mle import MaximumLikelihood
from .policy_gradient import DeterministicPolicyGradient
from .policy_gradient import ModelAwareDPG
from .policy_gradient import OneStepSVG
from .policy_gradient import ReparameterizedSoftPG


__all__ = [
    "ClippedDoubleQLearning",
    "SoftCDQLearning",
    "DeterministicPolicyGradient",
    "ModelAwareDPG",
    "OneStepSVG",
    "ReparameterizedSoftPG",
    "MaximumLikelihood",
    "DPGAwareModelLearning",
]
