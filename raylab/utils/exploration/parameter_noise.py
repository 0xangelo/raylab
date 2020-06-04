# pylint:disable=missing-module-docstring
from typing import Any
from typing import Optional
from typing import Tuple

import torch
from ray.rllib import SampleBatch
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.utils import override
from ray.rllib.utils.exploration import Exploration
from ray.rllib.utils.torch_ops import convert_to_non_torch_type

from raylab.policy import TorchPolicy
from raylab.pytorch.nn.distributions.flows import TanhSquashTransform
from raylab.pytorch.nn.utils import perturb_params
from raylab.utils.param_noise import AdaptiveParamNoiseSpec
from raylab.utils.param_noise import ddpg_distance_metric

from .random_uniform import RandomUniform


class ParameterNoise(RandomUniform):
    """Adds adaptive parameter noise exploration schedule to a Policy.

    Args:
        param_noise_spec: Arguments for `AdaptiveParamNoiseSpec`.
    """

    def __init__(self, *args, param_noise_spec: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        param_noise_spec = param_noise_spec or {}
        self._param_noise_spec = AdaptiveParamNoiseSpec(**param_noise_spec)
        self._squash = TanhSquashTransform(
            low=torch.as_tensor(self.action_space.low),
            high=torch.as_tensor(self.action_space.high),
        )

    @override(RandomUniform)
    def get_exploration_action(
        self,
        *,
        action_distribution: ActionDistribution,
        timestep: int,
        explore: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        model, inputs = action_distribution.model, action_distribution.inputs
        if explore:
            if timestep < self._pure_exploration_steps:
                return super().get_exploration_action(
                    action_distribution=action_distribution,
                    timestep=timestep,
                    explore=explore,
                )
            return model.behavior(**inputs), None
        return model.actor(**inputs), None

    @override(Exploration)
    def on_episode_start(
        self,
        policy: TorchPolicy,
        *,
        environment: Any = None,
        episode: Any = None,
        tf_sess: Any = None
    ):
        # pylint:disable=unused-argument
        perturb_params(
            policy.module.behavior,
            policy.module.actor,
            self._param_noise_spec.curr_stddev,
        )

    @torch.no_grad()
    @override(Exploration)
    def postprocess_trajectory(
        self, policy: TorchPolicy, sample_batch: SampleBatch, tf_sess: Any = None
    ):
        self.update_parameter_noise(policy, sample_batch)
        return sample_batch

    @override(Exploration)
    def get_info(self) -> dict:
        return {"param_noise_stddev": self._param_noise_spec.curr_stddev}

    def update_parameter_noise(self, policy: TorchPolicy, sample_batch: SampleBatch):
        """Update parameter noise stddev given a batch from the perturbed policy."""
        module = policy.module
        cur_obs = policy.convert_to_tensor(sample_batch[SampleBatch.CUR_OBS])
        actions = policy.convert_to_tensor(sample_batch[SampleBatch.ACTIONS])
        target_actions = module.actor(cur_obs)
        unsquashed_acts, _ = self._squash(actions, reverse=True)
        unsquashed_targs, _ = self._squash(target_actions, reverse=True)

        noisy, target = map(
            convert_to_non_torch_type, (unsquashed_acts, unsquashed_targs)
        )
        distance = ddpg_distance_metric(noisy, target)
        self._param_noise_spec.adapt(distance)
