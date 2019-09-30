# pylint: disable=missing-docstring
# pylint: enable=missing-docstring
from gym_cartpole_swingup.envs import CartPoleSwingUpEnv as _CartPoleSwingUpEnv


class CartPoleSwingUpEnv(_CartPoleSwingUpEnv):
    """CartPoleSwingUp task that exposes a differentiable PyTorch reward function."""

    @staticmethod
    def reward_fn(state, action, next_state):  # pylint: disable=unused-argument
        """
        Compute the reward function given a possibly batched transition.
        Assumes all but the last dimension are batch ones.
        """
        return next_state[..., 2]
