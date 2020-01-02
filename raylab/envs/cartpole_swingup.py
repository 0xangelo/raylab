# pylint: disable=missing-docstring
# pylint: enable=missing-docstring
import numpy as np
import torch
from gym_cartpole_swingup.envs import CartPoleSwingUpEnv as _CartPoleSwingUpEnv


class CartPoleSwingUpEnv(_CartPoleSwingUpEnv):
    """CartPoleSwingUp task that exposes a differentiable PyTorch reward function."""

    @staticmethod
    def _reward_fn(state, action, next_state):  # pylint: disable=unused-argument
        return (1 + np.cos(next_state.theta, dtype=np.float32)) / 2

    @staticmethod
    def reward_fn(state, action, next_state):  # pylint: disable=unused-argument
        """
        Compute the reward function given a possibly batched transition.
        Assumes all but the last dimension are batch ones.
        """
        return (1 + next_state[..., 2]) / 2

    def transition_fn(self, state, action, sample_shape=()):
        """Compute the next state and its log-probability.

        Accepts a `sample_shape` argument to sample multiple next states.
        """
        # pylint: disable=no-member,unused-argument
        action = action[..., 0] * self.params.forcemag

        xdot_update = self._calculate_xdot_update(state, action)
        thetadot_update = self._calculate_thetadot_update(state, action)

        delta_t = self.params.deltat
        new_x = state[..., 0] + state[..., 1] * delta_t
        new_xdot = state[..., 1] + xdot_update * delta_t
        new_costheta, new_sintheta = self._calculate_theta_update(state, delta_t)
        new_thetadot = state[..., 4] + thetadot_update * delta_t

        next_state = torch.stack(
            [new_x, new_xdot, new_costheta, new_sintheta, new_thetadot], dim=-1
        )
        return next_state.expand(sample_shape + state.shape), None

    def _calculate_xdot_update(self, state, action):
        # pylint: disable=no-member
        x_dot = state[..., 1]
        theta_dot = state[..., 4]
        cos_theta = state[..., 2]
        sin_theta = state[..., 3]
        return (
            -2 * self.params.mpl * (theta_dot ** 2) * sin_theta
            + 3 * self.params.pole.mass * self.params.gravity * sin_theta * cos_theta
            + 4 * action
            - 4 * self.params.friction * x_dot
        ) / (4 * self.params.masstotal - 3 * self.params.pole.mass * cos_theta ** 2)

    def _calculate_thetadot_update(self, state, action):
        # pylint: disable=no-member
        x_dot = state[..., 1]
        theta_dot = state[..., 4]
        cos_theta = state[..., 2]
        sin_theta = state[..., 3]
        return (
            -3 * self.params.mpl * (theta_dot ** 2) * sin_theta * cos_theta
            + 6 * self.params.masstotal * self.params.gravity * sin_theta
            + 6 * (action - self.params.friction * x_dot) * cos_theta
        ) / (
            4 * self.params.pole.length * self.params.masstotal
            - 3 * self.params.mpl * cos_theta ** 2
        )

    @staticmethod
    def _calculate_theta_update(state, delta_t):
        cos_theta = state[..., 2]
        sin_theta = state[..., 3]
        sin_theta_dot = torch.sin(state[..., 4] * delta_t)
        cos_theta_dot = torch.cos(state[..., 4] * delta_t)
        new_sintheta = sin_theta * cos_theta_dot + cos_theta * sin_theta_dot
        new_costheta = cos_theta * cos_theta_dot - sin_theta * sin_theta_dot
        return new_costheta, new_sintheta
