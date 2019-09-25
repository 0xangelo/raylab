# pylint: disable=missing-docstring
# pylint: enable=missing-docstring
import numpy as np
import torch
from gym_cartpole_swingup.envs import CartPoleSwingUpEnv as _CartPoleSwingUpEnv


class CartPoleSwingUpEnv(_CartPoleSwingUpEnv):
    """CartPoleSwingUp task that exposes a differentiable PyTorch reward function."""

    def reward_fn(self, state, action, next_state):  # pylint: disable=unused-argument
        """
        Compute the reward function given a possibly batched transition.
        Assumes all but the last dimension are batch ones.
        """
        reward_theta = (next_state[..., 2] + 1.0) / 2.0
        reward_x = torch.cos((next_state[..., 0] / self.x_threshold) * (np.pi / 2.0))
        return reward_theta * reward_x
