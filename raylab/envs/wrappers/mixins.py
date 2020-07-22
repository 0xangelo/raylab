"""Mixins for Gym environment wrappers."""
from typing import List
from typing import Optional

import gym.utils.seeding as seeding


class RNGMixin:
    """Adds a separate random number generator to an environment wrapper.

    Appends the wrapper's rng seed to the list of seeds returned by
    :meth:`seed`.

    Attributes:
        np_random: A numpy RandomState
    """

    # pylint:disable=missing-function-docstring,too-few-public-methods
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.np_random, _ = seeding.np_random()

    def seed(self, seed: Optional[int] = None) -> List[int]:
        seeds = super().seed(seed) or []
        self.np_random, seed_ = seeding.np_random(seed)
        return seeds + [seed_]
