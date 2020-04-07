# pylint:disable=missing-docstring
# pylint:enable=missing-docstring
from ray.rllib.utils.annotations import override
from ray.rllib.utils.exploration import Exploration


from .random_uniform import RandomUniformMixin


class StochasticActor(RandomUniformMixin, Exploration):
    """Exploration class compatible with StochasticActorMixin submodules."""

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
            return model.actor.sample(distribution_inputs)
        return model.actor.mode(distribution_inputs)
