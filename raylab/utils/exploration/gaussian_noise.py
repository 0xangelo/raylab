# pylint:disable=missing-module-docstring
from ray.rllib.utils import override

from .random_uniform import RandomUniform


class GaussianNoise(RandomUniform):
    """Adds fixed additive gaussian exploration noise to actions.



    Args:
        noise_stddev (float): Standard deviation of the Gaussian samples.
    """

    def __init__(self, *args, noise_stddev=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._noise_stddev = noise_stddev

    @override(RandomUniform)
    def get_exploration_action(self, *, action_distribution, timestep, explore=True):
        if explore:
            if timestep < self._pure_exploration_steps:
                return super().get_exploration_action(
                    action_distribution=action_distribution,
                    timestep=timestep,
                    explore=explore,
                )
            return action_distribution.sample_inject_noise(self._noise_stddev)
        return action_distribution.deterministic_sample()
