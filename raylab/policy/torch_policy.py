# pylint: disable=missing-docstring
# pylint: enable=missing-docstring
import os
import contextlib
from abc import ABC, abstractmethod

import torch
from ray.tune.logger import pretty_print
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.annotations import override
from ray.rllib.utils.tracking_dict import UsageTrackingDict
from ray.rllib.policy.policy import Policy, LEARNER_STATS_KEY
from ray.rllib.policy.sample_batch import SampleBatch

from raylab.utils.pytorch import convert_to_tensor


class TorchPolicy(ABC, Policy):
    """Custom TorchPolicy that aims to be more general than RLlib's one."""

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self._raw_user_config = config
        self.config = merge_dicts(self.get_default_config(), self._raw_user_config)
        self.device = (
            torch.device("cuda")
            if bool(os.environ.get("CUDA_VISIBLE_DEVICES", None))
            else torch.device("cpu")
        )
        self.module = self.make_module(observation_space, action_space, self.config)
        self.module.to(self.device)
        self._optimizer = self.optimizer()

    @staticmethod
    @abstractmethod
    def get_default_config():
        """Return the default config for this policy class."""

    @abstractmethod
    def make_module(self, obs_space, action_space, config):
        """Build the PyTorch nn.Module to be used by this policy.

        Arguments:
            obs_space (gym.spaces.Space): the observation space for this policy
            action_space (gym.spaces.Space): the action_space for this policy
            config (dict): the user config merged with the default one.

        Returns:
            A neural network module.
        """

    @abstractmethod
    def optimizer(self):
        """PyTorch optimizer to use."""

    @override(Policy)
    def get_weights(self):
        return {k: v.cpu() for k, v in self.module.state_dict().items()}

    @override(Policy)
    def set_weights(self, weights):
        self.module.load_state_dict(weights)

    @override(Policy)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        if not self.config["env_config"].get("time_aware"):
            hit_limit = sample_batch[SampleBatch.INFOS][-1].get("TimeLimit.truncated")
            sample_batch[SampleBatch.DONES][-1] = not hit_limit
        return sample_batch

    def convert_to_tensor(self, arr):
        """Convert an array to a PyTorch tensor in this policy's device."""
        return convert_to_tensor(arr, self.device)

    def _lazy_tensor_dict(self, sample_batch):
        tensor_batch = UsageTrackingDict(sample_batch)
        tensor_batch.set_get_interceptor(self.convert_to_tensor)
        return tensor_batch

    @staticmethod
    def _learner_stats(info):
        return {LEARNER_STATS_KEY: info}

    @contextlib.contextmanager
    def freeze_nets(self, *names):
        """Disable gradient requirements for the desired modules in this context."""
        try:
            for name in names:
                self.module[name].requires_grad_(False)
            yield
        finally:
            for name in names:
                self.module[name].requires_grad_(True)

    def __repr__(self):
        args = ["{name}(", "{observation_space}, ", "{action_space}, ", "{config}", ")"]
        config = pretty_print(self._raw_user_config).rstrip("\n")
        kwargs = dict(
            name=self.__class__.__name__,
            observation_space=self.observation_space,
            action_space=self.action_space,
        )

        if "\n" in config:
            config = "{\n" + config + "\n}"
            config = _addindent(config, 2)

            fmt = "\n".join(args)
            fmt = fmt.format(config=config, **kwargs)
            fmt = _addindent(fmt, 2)
        else:
            fmt = "".join(args)
            fmt = fmt.format(config=config, **kwargs)
        return fmt


def _addindent(tex_, num_spaces):
    tex = tex_.split("\n")
    # don't do anything for single-line stuff
    if len(tex) == 1:
        return tex_
    first = tex.pop(0)
    last = ""
    if len(tex) > 2:
        last = tex.pop()
    tex = [(num_spaces * " ") + line for line in tex]
    tex = "\n".join(tex)
    tex = first + "\n" + tex
    tex = tex + "\n" + last
    return tex
