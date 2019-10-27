# pylint: disable=missing-docstring
# pylint: enable=missing-docstring
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
        noisy_actions = self._compute_noisy_actions(sample_batch)
        target_actions = self._compute_noise_free_actions(sample_batch)
        distance = ddpg_distance_metric(noisy_actions, target_actions)
        self._param_noise_spec.adapt(distance)

    def perturb_policy_parameters(self):
        """Update the perturbed policy's parameters for exploration."""
        perturb_module_params(
            self.module["perturbed_policy"],
            self.module["policy"],
            self.curr_param_stddev,
        )

    @property
    def curr_param_stddev(self):
        """Return the current parameter noise standard deviation."""
        return self._param_noise_spec.curr_stddev

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

    def _compute_noise_free_actions(self, sample_batch):
        """Compute actions with the unperturbed policy.

        Override this for custom action computation.

        Arguments:
            sample_batch (SampleBatch): batch of current experience

        Returns:
            Actions as numpy arrays.
        """
        obs_tensors = self.convert_to_tensor(sample_batch[SampleBatch.CUR_OBS])
        return self.module["policy"](obs_tensors).numpy()

    def _compute_noisy_actions(self, sample_batch):  # pylint: disable=no-self-use
        """Compute actions with the perturbed policy.

        Override this for custom action computation.

        Arguments:
            sample_batch (SampleBatch): batch of current experience

        Returns:
            Actions as numpy arrays.
        """
        return sample_batch[SampleBatch.ACTIONS]
