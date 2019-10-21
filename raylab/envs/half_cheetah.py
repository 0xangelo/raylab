# pylint: disable=missing-docstring
# pylint: enable=missing-docstring
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv as _Base


class HalfCheetahEnv(_Base):
    """HalfCheetah Mujoco environment which exposes a differentiable reward function."""

    def __init__(self, **options):
        super().__init__(exclude_current_positions_from_observation=False, **options)

    def reward_fn(self, state, action, next_state):
        """Compute rewards given a possibly batched transition.

        Assumes all but the last dimension are batch ones.
        """
        x_position_before = state[..., 0]
        x_position_after = next_state[..., 0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        control_cost = self._ctrl_cost_weight * (action ** 2).sum(dim=-1)

        forward_reward = self._forward_reward_weight * x_velocity

        return forward_reward - control_cost
