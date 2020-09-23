"""Custom Gym wrappers for environments."""
from .correlated_irrelevant import CorrelatedIrrelevant
from .gaussian_random_walks import GaussianRandomWalks
from .linear_redundant import LinearRedundant
from .nonlinear_redundant import NonlinearRedundant
from .random_irrelevant import RandomIrrelevant
from .single_precision import SinglePrecision
from .time_aware_env import AddRelativeTimestep


__all__ = [
    "AddRelativeTimestep",
    "CorrelatedIrrelevant",
    "GaussianRandomWalks",
    "LinearRedundant",
    "NonlinearRedundant",
    "RandomIrrelevant",
    "SinglePrecision",
]
