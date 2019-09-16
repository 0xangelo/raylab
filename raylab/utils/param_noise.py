"""
From OpenAI Baselines:
https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
"""
from math import sqrt

import numpy as np


class AdaptiveParamNoiseSpec:
    """Adaptive schedule for parameter noise exploration.

    Note that initial_stddev and _stddev refer to std of parameter noise,
    but desired_action_stddev refers to (as name notes) desired std in action space.
    """

    def __init__(
        self, initial_stddev=0.1, desired_action_stddev=0.2, adaptation_coefficient=1.01
    ):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adaptation_coefficient = adaptation_coefficient

        self._stddev = initial_stddev

    def adapt(self, distance):
        """Update current stddev based on action space distance."""
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self._stddev /= self.adaptation_coefficient
        else:
            # Increase stddev.
            self._stddev *= self.adaptation_coefficient

    @property
    def stddev(self):
        """Return the current standard deviation."""
        return self._stddev

    def __repr__(self):
        fmt = (
            "AdaptiveParamNoiseSpec("
            "initial_stddev={}, desired_action_stddev={}, adaptation_coefficient={})"
        )
        return fmt.format(
            self.initial_stddev, self.desired_action_stddev, self.adaptation_coefficient
        )


def ddpg_distance_metric(actions1, actions2):
    """Compute "distance" between actions taken by two policies at the same states.

    Expects numpy arrays.
    """
    diff = actions1 - actions2
    mean_diff = np.mean(np.square(diff), axis=0)
    dist = sqrt(np.mean(mean_diff))
    return dist
