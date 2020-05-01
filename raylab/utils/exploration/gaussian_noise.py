# pylint:disable=missing-module-docstring
import torch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.exploration import Exploration

from raylab.modules.distributions import TanhSquashTransform

from .random_uniform import RandomUniformMixin


class GaussianNoise(RandomUniformMixin, Exploration):
    """Adds fixed additive gaussian exploration noise to actions before squashing.

    Args:
        noise_stddev (float): Standard deviation of the Gaussian samples.
    """

    def __init__(self, *args, noise_stddev=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._noise_stddev = noise_stddev
        self._squash = TanhSquashTransform(
            low=torch.as_tensor(self.action_space.low),
            high=torch.as_tensor(self.action_space.high),
        )

    @override(Exploration)
    def get_exploration_action(
        self, distribution_inputs, action_dist_class, model, timestep, explore=True,
    ):
        # pylint:disable=too-many-arguments
        if explore:
            if timestep < self._pure_exploration_steps:
                return super().get_exploration_action(
                    distribution_inputs, action_dist_class, model, timestep, explore
                )
            actions = model.actor(distribution_inputs)
            pre_squash, _ = self._squash(actions, reverse=True)
            noise = torch.randn_like(pre_squash) * self._noise_stddev
            return self._squash(pre_squash + noise)[0], None
        return model.actor(distribution_inputs), None
