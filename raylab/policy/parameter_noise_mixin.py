# pylint: disable=missing-docstring
# pylint: enable=missing-docstring
from abc import abstractmethod

import torch
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override

from raylab.utils.pytorch import perturb_module_params
from raylab.utils.param_noise import AdaptiveParamNoiseSpec, ddpg_distance_metric
from .torch_policy import TorchPolicy


class AdaptiveParamNoiseMixin:
    """Adds adaptive parameter noise exploration schedule to a Policy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config["exploration"] == "parameter_noise":
            self._param_noise_spec = AdaptiveParamNoiseSpec(
                **self.config["param_noise_spec"]
            )

    def update_parameter_noise(self, sample_batch):
        """Update parameter noise stddev given a batch from the perturbed policy."""
        noisy_actions = sample_batch[SampleBatch.ACTIONS]
        target_actions = self._compute_noise_free_actions(
            sample_batch[SampleBatch.CUR_OBS]
        )
        distance = ddpg_distance_metric(noisy_actions, target_actions)
        self._param_noise_spec.adapt(distance)

    def perturb_policy_parameters(self):
        """Update the perturbed policy's parameters for exploration."""
        perturb_module_params(
            self.module["policy"],
            self.module["target_policy"],
            self._param_noise_spec.curr_stddev,
        )

    @torch.no_grad()
    @override(TorchPolicy)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):  # pylint: disable=missing-docstring
        sample_batch = super().postprocess_trajectory(
            sample_batch, other_agent_batches=other_agent_batches, episode=episode
        )  # pylint: disable=no-member
        if self.config["exploration"] == "parameter_noise":
            self.update_parameter_noise(sample_batch)
        return sample_batch

    @abstractmethod
    def _compute_noise_free_actions(self, obs_batch):
        """Compute actions with the unperturbed policy.

        Arguments:
            obs_batch (np.ndarray): batch of current observations

        Returns:
            Actions as numpy arrays.
        """
