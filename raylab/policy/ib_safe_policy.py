"""For exploring the Industrial Benchmark."""
import numpy as np
from ray.rllib.policy import Policy
from ray.rllib.utils.annotations import override


class IBSafePolicy(Policy):
    """Behaviour policy for collection data in IndustrialBenchmark.

    Based on the policy described in Appendix C.3.1 of
    https://arxiv.org/abs/1710.07283
    """

    # pylint: disable=abstract-method

    @override(Policy)
    def compute_actions(
        self,
        obs_batch,
        state_batches,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs
    ):
        # pylint: disable=too-many-arguments,too-many-locals
        obs_batch = np.asarray(obs_batch)
        batch_size = len(obs_batch)
        z_v, z_g = np.clip(
            np.array([-1.0, -1.0]),
            np.array([1.0, 1.0]),
            np.random.normal(loc=0.5, scale=1 / np.sqrt(3), size=(batch_size, 2)),
        ).T
        u_v, u_g, u_s = np.random.uniform(low=-1.0, high=1.0, size=(batch_size, 3)).T
        # Equation (15)
        delta_v = np.where(
            obs_batch[..., 1] < 40, z_v, np.where(obs_batch[..., 1] > 60, -z_v, u_v)
        )
        # Equation (16)
        delta_g = np.where(
            obs_batch[..., 2] < 40, z_g, np.where(obs_batch[..., 2] > 60, -z_g, u_g)
        )
        actions = np.array((delta_v, delta_g, u_s)).T
        return actions, state_batches, {}
