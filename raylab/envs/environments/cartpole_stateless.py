"""Partially observed variant of the CartPole gym environment.

https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

We delete the velocity component of the state, so that it can only be solved
by an LSTM policy.
"""
import gym
import numpy as np
from gym import spaces
from gym.envs.classic_control.cartpole import CartPoleEnv


class CartPoleStateless(gym.ObservationWrapper):
    """Partially observed variant of the CartPole gym environment."""

    def __init__(self):
        super().__init__(CartPoleEnv)
        high = np.r_[
            self.env.observation_space.high[0], self.env.observation_space.high[2]
        ]
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def observation(self, observation):
        return np.r_[observation[0], observation[2]]
