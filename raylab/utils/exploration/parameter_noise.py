# pylint:disable=missing-module-docstring
from typing import Any
from typing import Optional
from typing import Tuple

import torch
from ray.rllib import SampleBatch
from ray.rllib.models.action_dist import ActionDistribution

from raylab.policy import TorchPolicy
from raylab.pytorch.nn.utils import perturb_params
from raylab.utils.param_noise import AdaptiveParamNoiseSpec
from raylab.utils.param_noise import ddpg_distance_metric

from .base import Model
from .random_uniform import RandomUniform


class ParameterNoise(RandomUniform):
    """Adds adaptive parameter noise exploration schedule to a Policy.

    Expects `actor` attribute of `policy.module` to be an instance of
    `raylab.policy.modules.actor.policy.deterministic.DeterministicPolicy`.

    Args:
        param_noise_spec: Arguments for `AdaptiveParamNoiseSpec`.
    """

    def __init__(self, *args, param_noise_spec: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        param_noise_spec = param_noise_spec or {}
        self._param_noise_spec = AdaptiveParamNoiseSpec(**param_noise_spec)

    def get_exploration_action(
        self,
        *,
        action_distribution: ActionDistribution,
        timestep: int,
        explore: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if explore:
            if timestep < self._pure_exploration_steps:
                return super().get_exploration_action(
                    action_distribution=action_distribution,
                    timestep=timestep,
                    explore=explore,
                )
            return action_distribution.sample()
        return action_distribution.deterministic_sample()

    def on_episode_start(
        self,
        policy: TorchPolicy,
        *,
        environment: Any = None,
        episode: Any = None,
        tf_sess: Any = None,
    ):
        # pylint:disable=unused-argument
        perturb_params(
            policy.module.behavior,
            policy.module.actor,
            self._param_noise_spec.curr_stddev,
        )

    @torch.no_grad()
    def postprocess_trajectory(
        self, policy: TorchPolicy, sample_batch: SampleBatch, tf_sess: Any = None
    ):
        self.update_parameter_noise(policy, sample_batch)
        return sample_batch

    def get_info(self, sess=None) -> dict:
        return {"param_noise_stddev": self._param_noise_spec.curr_stddev}

    def update_parameter_noise(self, policy: TorchPolicy, sample_batch: SampleBatch):
        """Update parameter noise stddev given a batch from the perturbed policy."""
        module = policy.module
        cur_obs = policy.convert_to_tensor(sample_batch[SampleBatch.CUR_OBS])
        actions = policy.convert_to_tensor(sample_batch[SampleBatch.ACTIONS])

        noisy = module.actor.unsquash_action(actions)
        target = module.actor.unconstrained_action(cur_obs)
        noisy, target = map(lambda x: x.cpu().detach().numpy(), (noisy, target))

        distance = ddpg_distance_metric(noisy, target)
        self._param_noise_spec.adapt(distance)

    @classmethod
    def check_model_compat(cls, model: Model):
        assert (
            model is not None
        ), f"Need to pass the model to {cls} to check compatibility."
        actor, behavior = model.actor, model.behavior
        assert set(actor.parameters()).isdisjoint(set(behavior.parameters())), (
            "Target and behavior policy cannot share parameters in parameter "
            "noise exploration."
        )
