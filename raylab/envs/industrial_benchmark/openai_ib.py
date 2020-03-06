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
import torch
import gym
from gym import spaces

from .ids import IDS
from .dynamics import fatigue, op_cost
from .goldstone.torch.environment import TorchGSEnvironment

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
        miscalibration=True,
    ):
        # pylint:disable=too-many-arguments
        # Setting up the IB environment
        self.setpoint = setpoint
        self._ib = IDS(setpoint, miscalibration=miscalibration)
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
            state=np.around(self._ib.visible_state()[1:4], 0),
            action=action,
        )

        # reward is divided by 100 to improve learning
        return self._get_obs(), return_reward / 100, False, self.minimal_markov_state()

    def _get_obs(self):
        if self.observation == "visible":
            obs = self._ib.visible_state()
        elif self.observation == "markovian":
            obs = self._ib.minimal_markov_state()
        elif self.observation == "full":
            obs = self._ib.full_state()
        return obs.astype(np.float32)

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
        return dict(zip(markovian_state_variables, self._ib.minimal_markov_state()))

    def reward_fn(self, state, _, next_state):
        """Compute the current reward according to equation (5) of the paper."""
        fat_coeff, con_coeff = self._ib.CRF, self._ib.CRC
        fatigue_ = next_state[..., 4]
        consumption = next_state[..., 5]
        reward = -(fat_coeff * fatigue_ + con_coeff * consumption)
        if self.reward_function == "delta":
            old_fat = state[..., 4]
            old_con = state[..., 5]
            reward = reward + fat_coeff * old_fat + con_coeff * old_con
        return reward / 100

    def transition_fn(self, state, action, sample_shape=()):
        """Compute the next state and its log-probability."""

        # Expand state, action
        state = state.expand(torch.Size(sample_shape) + state.shape)
        action = action.expand(torch.Size(sample_shape) + action.shape)
        # Get operational cost ommitted from the cost history
        coc = op_cost.current_operational_cost(state)
        # addAction
        next_state, effective_shift = self._add_action(state, action)
        # update fatigue
        next_state = self._update_fatigue(next_state)
        # update consumption
        next_state = self._update_consumption(next_state, coc, effective_shift)
        return next_state, None

    def _add_action(self, state, action):
        # pylint:disable=protected-access
        setpoint, velocity, gain, shift = state[..., :4].chunk(4, dim=-1)
        delta_v, delta_g, delta_h = action.chunk(3, dim=-1)

        velocity = torch.clamp(velocity + delta_v, 0.0, 100.0)
        gain = torch.clamp(gain + 10 * delta_g, 0.0, 100.0)
        shift = torch.clamp(
            shift
            + ((self._ib.max_required_step / 0.9) * 100.0 / self._ib.gs_scale)
            * delta_h,
            0.0,
            100.0,
        )

        if self._ib._miscalibration:
            effective_shift = torch.clamp(
                self._ib.gs_scale * shift / 100.0
                - self._ib.gs_setpoint_dependency * setpoint
                - self._ib.gs_bound,
                -self._ib.gs_bound,
                self._ib.gs_bound,
            )
        else:
            phi = state[..., -3:-2]
            effective_shift = torch.sin(np.pi * phi / 12)

        next_state = torch.cat(
            [setpoint, velocity, gain, shift, state[..., 4:]], dim=-1
        )
        return next_state, effective_shift

    @staticmethod
    def _update_fatigue(state):
        """
        The sub-dynamics of fatigue are influenced by the same
        variables as the sub-dynamics of operational cost, i.e., setpoint p, velocity v,
        and gain g. The IB is designed in such a way that, when changing the steerings
        velocity v and gain g as to reduce the operational cost, fatigue will be
        increased, leading to the desired multi-criterial task, with two reward
        components showing opposite dependencies on the actions.
        """
        setpoint, velocity, gain = state[..., :3].chunk(3, dim=-1)
        hidden_velocity, hidden_gain = state[..., -2:].chunk(2, dim=-1)

        # Equations (26, 27)
        eff_velocity = fatigue.effective_velocity(velocity, gain, setpoint)
        eff_gain = fatigue.effective_gain(gain, setpoint)

        # Equations (28, 29)
        noise_velocity, noise_gain = fatigue.sample_noise_variables(
            eff_velocity, eff_gain
        )

        # Equation (30)
        hidden_velocity = fatigue.update_hidden_velocity(
            hidden_velocity, eff_velocity, noise_velocity
        )
        # Equation (31)
        hidden_gain = fatigue.update_hidden_gain(hidden_gain, eff_gain, noise_gain)

        # Equation (23)
        alpha = fatigue.sample_alpha(
            hidden_velocity, noise_velocity, hidden_gain, noise_gain
        )
        # Equations (21, 22)
        new_fatigue = fatigue.fatigue(fatigue.basic_fatigue(velocity, gain), alpha)

        return torch.cat(
            [
                state[..., :4],
                new_fatigue,
                state[..., 5:-2],
                hidden_velocity,
                hidden_gain,
            ],
            dim=-1,
        )

    def _update_consumption(self, state, coc, effective_shift):
        """Dynamics of operational cost."""
        state, conv_cost = self._update_operational_cost(state, coc)
        state, miscalibration = self._update_miscalibration(state, effective_shift)

        # This seems to correspond to equation (19),
        # although the minus sign is mysterious.
        # ct_hat = conv_cost - (self._ib.CRGS * (miscalibration - 1.0))
        ct_hat = conv_cost + self._ib.CRGS * miscalibration
        # This corresponds to equation (20), although the constant 0.005 is
        # different from the 0.02 written in the paper. This might result in
        # very low observational noise
        # consumption = ct_hat - torch.randn_like(ct_hat) * (1 + 0.005 * ct_hat)
        consumption = ct_hat + torch.randn_like(ct_hat) * (1 + 0.02 * ct_hat)
        return torch.cat([state[..., :5], consumption, state[..., 6:]], dim=-1)

    @staticmethod
    def _update_operational_cost(state, coc):
        """
        The sub-dynamics of operational cost are influenced by the external driver
        setpoint p and two of the three steerings, velocity v and gain g.
        """
        state = op_cost.update_operational_cost_history(state, coc)
        conv_cost = op_cost.convoluted_operational_cost(state)
        return state, conv_cost

    def _update_miscalibration(self, state, effective_shift):
        """Calculate the domain, response, direction, and MisCalibration.

        Computes equations (9), (10), (11), and (12) of the paper under the
        'Dynamics of mis-calibration' section.
        """
        domain, system_response, phi_idx = state[..., -5:-2].chunk(3, dim=-1)
        gs_env = TorchGSEnvironment(
            24, self._ib.max_required_step, self._ib.max_required_step / 2.0
        )

        reward, domain, phi_idx, system_response = gs_env.state_transition(
            domain.sign(), phi_idx, system_response.sign(), effective_shift
        )
        next_state = torch.cat(
            [state[..., :-5], domain, system_response, phi_idx, state[..., -2:]], dim=-1
        )
        miscalibration = -reward
        return next_state, miscalibration
