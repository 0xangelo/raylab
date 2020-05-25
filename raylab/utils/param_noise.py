"""
From OpenAI Baselines:
https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
"""
from dataclasses import dataclass
from dataclasses import field

import numpy as np


@dataclass
class AdaptiveParamNoiseSpec:
    """Adaptive schedule for parameter noise exploration.

    Note that initial_stddev and curr_stddev refer to std of parameter noise,
    but desired_action_stddev refers to (as name suggests) the desired stddev
    in action space.
    """

    initial_stddev: float = 0.1
    desired_action_stddev: float = 0.2
    adaptation_coeff: float = 1.01
    curr_stddev: float = field(init=False)

    def __post_init__(self):
        self.curr_stddev = self.initial_stddev

    def adapt(self, distance):
        """Update current stddev based on action space distance."""
        if distance > self.desired_action_stddev:
            self.curr_stddev /= self.adaptation_coeff  # Decrease stddev.
        else:
            self.curr_stddev *= self.adaptation_coeff  # Increase stddev.


def ddpg_distance_metric(actions1, actions2):
    """Compute "distance" between actions taken by two policies at the same states.

    Expects numpy arrays.
    """
    diff = actions1 - actions2
    mean_diff = np.mean(np.square(diff), axis=0)
    dist = np.sqrt(np.mean(mean_diff))
    return dist
