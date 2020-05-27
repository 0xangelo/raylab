# pylint:disable=missing-module-docstring
from ray.rllib.utils import override

from .random_uniform import RandomUniform


class StochasticActor(RandomUniform):
    """Exploration class compatible with StochasticActorMixin submodules."""

    @override(RandomUniform)
    def get_exploration_action(self, *, action_distribution, timestep, explore=True):
        # pylint:disable=too-many-arguments
        if explore:
            if timestep < self._pure_exploration_steps:
                return super().get_exploration_action(
                    action_distribution=action_distribution,
                    timestep=timestep,
                    explore=explore,
                )
            return action_distribution.sample()
        return action_distribution.deterministic_sample()
