# pylint:disable=missing-module-docstring
import numpy as np
from ray.rllib.utils.annotations import override
from ray.rllib.utils.exploration import Exploration

import raylab.utils.pytorch as ptu


class RandomUniform(Exploration):
    """Samples actions from the Gym action space

    Args:
        pure_exploration_steps (int): Number of initial timesteps to explore.
    """

    # pylint:disable=too-few-public-methods

    def __init__(self, *args, pure_exploration_steps=0, **kwargs):
        super().__init__(*args, **kwargs)
        if self.framework != "torch":
            raise ValueError(
                f"{type(self)} incompatible with '{self.framework}' framework."
            )
        self._pure_exploration_steps = pure_exploration_steps

    @override(Exploration)
    def get_exploration_action(self, *, action_distribution, timestep, explore=True):
        # pylint:disable=unused-argument
        if explore:
            obs = action_distribution.inputs["obs"]
            acts = ptu.convert_to_tensor(
                [self.action_space.sample() for _ in range(obs.size(0))], obs.device
            )
            logp = ptu.convert_to_tensor(
                [-np.log(self.action_space.high - self.action_space.low).sum(axis=-1)]
                * obs.size(0),
                obs.device,
            )
            return acts, logp
        return action_distribution.deterministic_sample()
