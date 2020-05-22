"""Modularized losses/procedures for RL algorithms."""

from .cdq_learning import ClippedDoubleQLearning
from .cdq_learning import SoftCDQLearning
from .policy_gradient import DeterministicPolicyGradient
from .policy_gradient import ModelAwareDPG
from .policy_gradient import ReparameterizedSoftPG
