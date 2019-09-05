"""Custom Replay Buffer subclassing RLlibs's implementation."""
from ray.rllib.utils.annotations import override
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.optimizers.replay_buffer import ReplayBuffer as _ReplayBuffer


class ReplayBuffer(_ReplayBuffer):
    """Replay buffer that returns a SampleBatch object when queried for samples."""

    @override(_ReplayBuffer)
    def sample(self, batch_size):
        obs_t, action, reward, obs_tp1, done = super().sample(batch_size)
        return SampleBatch(
            {
                SampleBatch.CUR_OBS: obs_t,
                SampleBatch.ACTIONS: action,
                SampleBatch.REWARDS: reward,
                SampleBatch.NEXT_OBS: obs_tp1,
                SampleBatch.DONES: done,
            }
        )
