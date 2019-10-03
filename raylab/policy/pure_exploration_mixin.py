# pylint: disable=missing-docstring
# pylint: enable=missing-docstring
import torch


class PureExplorationMixin:
    """Adds uniform random action sampling to a Policy when enabled."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pure_exploration = False

    def set_pure_exploration_phase(self, phase):
        """Set a boolean flag that tells the policy to act randomly."""
        self._pure_exploration = phase

    @property
    def is_uniform_random(self):
        """Return whether this policy is sampling actions uniformly at random."""
        return self._pure_exploration

    def _uniform_random_actions(self, obs_batch):
        dist = torch.distributions.Uniform(
            self.convert_to_tensor(self.action_space.low),
            self.convert_to_tensor(self.action_space.high),
        )
        actions = dist.sample(sample_shape=obs_batch.shape[:-1])
        return actions
