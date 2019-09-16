# pylint: disable=missing-docstring
# pylint: enable=missing-docstring
import os
from abc import abstractmethod

import torch
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.annotations import override
from ray.rllib.utils.tracking_dict import UsageTrackingDict
from ray.rllib.policy import Policy

from raylab.utils.pytorch import convert_to_tensor


class TorchPolicy(Policy):
    """Custom TorchPolicy that aims to be more general than RLlib's one."""

    module = None

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self.config = merge_dicts(self.get_default_config(), config)
        self.device = (
            torch.device("cuda")
            if bool(os.environ.get("CUDA_VISIBLE_DEVICES", None))
            else torch.device("cpu")
        )

    @staticmethod
    @abstractmethod
    def get_default_config():
        """Return the default config for this policy class."""

    @override(Policy)
    def get_weights(self):
        return {k: v.cpu() for k, v in self.module.state_dict().items()}

    @override(Policy)
    def set_weights(self, weights):
        self.module.load_state_dict(weights)

    def convert_to_tensor(self, arr):
        """Convert an array to a PyTorch tensor in this policy's device."""
        return convert_to_tensor(arr, self.device)

    def _lazy_tensor_dict(self, sample_batch):
        tensor_batch = UsageTrackingDict(sample_batch)
        tensor_batch.set_get_interceptor(self.convert_to_tensor)
        return tensor_batch
