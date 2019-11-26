"""
The MIT License (MIT)

Copyright 2017 Technical University of Berlin

Authors: Ludwig Winkler

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import logging

import numpy as np
import gym
from gym import spaces

from .ids import IDS


logger = logging.getLogger(__name__)


class IBEnv(gym.Env):
    """OpenAI Gym wrapper for Industrial Benchmark.

    Currently only supports a fixed setpoint.
    """

    # pylint: disable=missing-docstring,too-many-instance-attributes

    def __init__(
        self,
        setpoint=50,
        reward_type="classic",
        action_type="continuous",
        observation="visible",
    ):
        # Setting up the IB environment
        self.setpoint = setpoint
        self._ib = IDS(setpoint)
        # Used to determine whether to return the absolute value or the relative change
        # in the cost function
        self.reward_function = reward_type
        self.action_type = action_type
        self.observation = observation

        # Observation bounds for visible state
        # [setpoint, velocity, gain, shift, fatigue, consumption]
        ob_low = np.array([0, 0, 0, 0, -np.inf, -np.inf])
        ob_high = np.array([100, 100, 100, 100, np.inf, np.inf])
        if self.observation == "markovian":
            # Observation bounds for minimal markovian state
            ob_low = np.concatenate([ob_low, np.array([-np.inf] * 14)])
            ob_high = np.concatenate([ob_high, np.array([np.inf] * 14)])
        elif self.observation == "full":
            # Observation bounds for full state
            ob_low = np.concatenate([ob_low, np.array([-np.inf] * 24)])
            ob_high = np.concatenate([ob_high, np.array([np.inf] * 24)])
        self.observation_space = spaces.Box(low=ob_low, high=ob_high, dtype=np.float32)

        # Action space and the observation space
        if self.action_type == "discrete":
            # Discrete action space with three different values per steerings for the
            # three steerings ( 3^3 = 27)
            self.action_space = spaces.Discrete(27)

            # A list of all possible actions discretized into [-1,0,1]
            # e.g. [[-1,-1,-1],[-1,-1,0],[-1,-1,1],[-1,0,-1],[-1,0,0], ... ]
            # Network has 27 outputs and chooses one environmental action out of the
            # discretized 27 possible actions
            self.env_action = []
            for vel in [-1, 0, 1]:
                for gain in [-1, 0, 1]:
                    for shift in [-1, 0, 1]:
                        self.env_action.append([vel, gain, shift])
        elif self.action_type == "continuous":
            # Continuous action space for each steering [-1,1]
            ac_low = np.array([-1, -1, -1])
            self.action_space = spaces.Box(ac_low, -ac_low, dtype=np.float32)
        else:
            raise ValueError(
                "Invalid action type {}. "
                "`action_type` can either be 'discrete' or 'continuous'".format(
                    self.action_type
                )
            )

        self.reward = -self._ib.state["cost"]
        # Alternative reward that returns the improvement or decrease in the cost
        # If the cost function improves/decreases, the reward is positiv
        # If the cost function deteriorates/increases, the reward is negative
        # e.g.: -400 -> -450 = delta_reward of -50
        self.delta_reward = 0

        # smoothed_cost is used as a smoother cost function for monitoring the agent
        # & environment with lower variance
        # Updates itself with 0.95*old_cost + 0.05*new_cost or any other linear
        # combination
        self.smoothed_cost = self._ib.state["cost"]

        self.seed()

    def step(self, action):
        # Executing the action and saving the observation
        if self.action_type == "discrete":
            self._ib.step(self.env_action[action])
        elif self.action_type == "continuous":
            self._ib.step(action)

        # Calculating both the relative reward (improvement or decrease) and updating
        # the reward
        self.delta_reward = self.reward - self._ib.state["cost"]
        self.reward = self._ib.state["cost"]

        # Due to the very high stochasticity a smoothed cost function is easier to
        # follow visually
        self.smoothed_cost = int(
            0.9 * self.smoothed_cost + 0.1 * self._ib.state["cost"]
        )

        # Two reward functions are available:
        # 'classic' which returns the original cost and
        # 'delta' which returns the change in the cost function w.r.t. the previous cost
        if self.reward_function == "classic":
            return_reward = -self._ib.state["cost"]
        elif self.reward_function == "delta":
            return_reward = self.delta_reward
        else:
            raise ValueError(
                "Invalid reward function specification. Use 'classic' for the original "
                "cost function or 'delta' for the change in the cost fucntion between "
                "steps."
            )

        logger.info(
            "Cost smoothed: %(cost)s, State (v, g, s): %(state)s, Action: %(action)s",
            cost=-self.smoothed_cost,
            state=np.around(self._ib.visibleState()[1:4], 0),
            action=action,
        )

        # reward is divided by 100 to improve learning
        return self._get_obs(), return_reward / 100, False, self.minimal_markov_state()

    def _get_obs(self):
        if self.observation == "visible":
            obs = self._ib.visibleState()
        elif self.observation == "markovian":
            obs = self._ib.minimalMarkovState()
        elif self.observation == "full":
            obs = self._ib.fullState()
        return obs.astype(np.float32)

    def reward_fn(self, state, action, next_state):  # pylint: disable=unused-argument
        """Compute the current reward according to equation (5) of the paper."""
        fat_coeff, con_coeff = self._ib.CRF, self._ib.CRC
        reward = -(fat_coeff * next_state[..., 4] + con_coeff * next_state[..., 5])
        if self.reward_function == "delta":
            reward = reward + fat_coeff * state[..., 4] + con_coeff * state[..., 5]
        return reward / 100

    def reset(self):
        # Resetting the entire environment
        self._ib.reset()
        self.reward = -self._ib.state["cost"]
        return self._get_obs()

    def seed(self, seed=None):
        return self._ib.set_seed(seed)

    def render(self, mode="human"):
        pass

    def minimal_markov_state(self):
        markovian_state_variables = [
            "setpoint",
            "velocity",
            "gain",
            "shift",
            "fatigue",
            "consumption",
            "op_cost_t1",
            "op_cost_t2",
            "op_cost_t3",
            "op_cost_t4",
            "op_cost_t5",
            "op_cost_t6",
            "op_cost_t7",
            "op_cost_t8",
            "op_cost_t9",
            "ml1",
            "ml2",
            "ml3",
            "hv",
            "hg",
        ]
        return dict(zip(markovian_state_variables, self._ib.minimalMarkovState()))
