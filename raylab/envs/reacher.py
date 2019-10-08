# pylint: disable=missing-docstring
# pylint: enable=missing-docstring
import torch
from gym.envs.mujoco.reacher import ReacherEnv as _ReacherEnv


class ReacherEnv(_ReacherEnv):
    """Reacher Mujoco environment which exposes its differentiable reward function."""

    @staticmethod
    def reward_fn(state, action, next_state):  # pylint: disable=unused-argument
        """Compute rewards given a possibly batched transition.

        Assumes all but the last dimension are batch ones.
        """
        dist = state[..., -3:]
        reward_dist = -torch.norm(dist, dim=-1)
        reward_ctrl = -torch.sum(action ** 2, dim=-1)
        return reward_dist + reward_ctrl
