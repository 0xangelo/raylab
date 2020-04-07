# pylint:disable=missing-docstring
# pylint:enable=missing-docstring
import numpy as np
from ray.rllib.utils.annotations import override
from ray.rllib.utils.exploration import Exploration


class RandomUniformMixin:
    """Samples actions from the Gym action space

    Args:
        pure_exploration_steps (int): Number of initial timesteps to explore.
    """

    # pylint:disable=too-few-public-methods

    def __init__(self, *args, pure_exploration_steps=0, **kwargs):
        super().__init__(*args, **kwargs)
        self._pure_exploration_steps = pure_exploration_steps

    @override(Exploration)
    def get_exploration_action(
        self, distribution_inputs, action_dist_class, model, timestep, explore=True,
    ):
        # pylint:disable=too-many-arguments,missing-docstring,unused-argument
        if explore:
            obs = distribution_inputs
            acts = [self.action_space.sample() for _ in range(obs.size(0))]
            logp = [
                -np.log(self.action_space.high - self.action_space.low).sum(axis=-1)
            ] * obs.size(0)
            return acts, logp
        return distribution_inputs, None
