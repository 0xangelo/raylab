# pylint:disable=missing-module-docstring
import numpy as np


def explained_variance(targets, pred):
    """Compute the explained variance given targets and predictions."""
    targets_var = np.var(targets, axis=0)
    diff_var = np.var(targets - pred, axis=0)
    return np.maximum(-1.0, 1.0 - (diff_var / targets_var))
