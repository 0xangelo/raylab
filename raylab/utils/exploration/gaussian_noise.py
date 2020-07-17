# pylint:disable=missing-module-docstring
from typing import Tuple

import torch
from torch import Tensor

from raylab.policy.action_dist import BaseActionDist
from raylab.policy.modules.actor.policy.deterministic import DeterministicPolicy

from .base import Model
from .random_uniform import RandomUniform


class GaussianNoise(RandomUniform):
    """Adds fixed additive gaussian exploration noise to actions.

    Args:
        noise_stddev: Standard deviation of the Gaussian samples
    """

    valid_behavior_cls = DeterministicPolicy

    def __init__(self, *args, noise_stddev: float = 0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self._noise_stddev = noise_stddev

    def get_exploration_action(
        self,
        *,
        action_distribution: BaseActionDist,
        timestep: int,
        explore: bool = True,
    ):
        if explore:
            if timestep < self._pure_exploration_steps:
                return super().get_exploration_action(
                    action_distribution=action_distribution,
                    timestep=timestep,
                    explore=explore,
                )
            return self._inject_gaussian_noise(action_distribution)
        return action_distribution.deterministic_sample()

    def _inject_gaussian_noise(
        self, action_distribution: BaseActionDist
    ) -> Tuple[Tensor, None]:
        model, inputs = action_distribution.model, action_distribution.inputs
        unconstrained_action = model.behavior.unconstrained_action(**inputs)
        unconstrained_action += (
            torch.randn_like(unconstrained_action) * self._noise_stddev
        )
        action = model.behavior.squash_action(unconstrained_action)
        return action, None

    @classmethod
    def check_model_compat(cls, model: Model):
        RandomUniform.check_model_compat(model)
        assert model is not None, f"{cls} exploration needs access to the NN."
        assert hasattr(
            model, "behavior"
        ), f"NN model {type(model)} has no behavior attribute."
        assert isinstance(model.behavior, cls.valid_behavior_cls), (
            f"Expected behavior to be an instance of {cls.valid_behavior_cls};"
            " found {type(model.behavior)} instead."
        )
