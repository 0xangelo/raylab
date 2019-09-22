# pylint: disable=missing-docstring
# pylint: enable=missing-docstring
import gym
import numpy as np


class AddRelativeTimestep(gym.ObservationWrapper):
    """
    Adds the relative timestep (normalized to the range [0, 1]) to the
    observations of environments with observation spaces of type gym.spaces.Box.
    """

    def __init__(self, env=None):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=np.append(self.observation_space.low, 0.0),
            high=np.append(self.observation_space.high, 1.0),
            dtype=self.observation_space.dtype,
        )

    def observation(self, observation):
        # pylint: disable=protected-access
        return np.append(
            observation, (self.env._elapsed_steps / self.spec.timestep_limit)
        )
