# pylint:disable=missing-module-docstring
import torch
from ray.rllib.utils.annotations import override

from raylab.modules.distributions import TanhSquashTransform

from .random_uniform import RandomUniform


class GaussianNoise(RandomUniform):
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

    @override(RandomUniform)
    def get_exploration_action(self, *, action_distribution, timestep, explore=True):
        if explore:
            if timestep < self._pure_exploration_steps:
                return super().get_exploration_action(
                    action_distribution=action_distribution,
                    timestep=timestep,
                    explore=explore,
                )
            return self._get_gaussian_perturbed_actions(action_distribution)
        return action_distribution.deterministic_sample()

    def _get_gaussian_perturbed_actions(self, action_distribution):
        module, inputs = action_distribution.model, action_distribution.inputs
        actions = module.actor(**inputs)
        pre_squash, _ = self._squash(actions, reverse=True)
        noise = torch.randn_like(pre_squash) * self._noise_stddev
        return self._squash(pre_squash + noise)[0], None
