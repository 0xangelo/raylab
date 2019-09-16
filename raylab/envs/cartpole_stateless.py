"""Partially observed variant of the CartPole gym environment.

https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

We delete the velocity component of the state, so that it can only be solved
by a LSTM policy."""


import gym
from gym import spaces
import numpy as np


class CartPoleStatelessWrapper(gym.ObservationWrapper):
    """Removes velocities from the state vector.

    This wrapper is specific to CartPoleEnv.
    """

    def __init__(self, env):
        super(CartPoleStatelessWrapper, self).__init__(env)
        high = np.r_[
            self.env.observation_space.high[0], self.env.observation_space.high[2]
        ]
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def observation(self, observation):
        # pylint: disable=missing-docstring
        return np.r_[observation[0], observation[2]]
