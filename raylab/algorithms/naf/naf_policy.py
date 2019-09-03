"""NAF policy class using PyTorch."""
import os
import inspect

import torch
from ray.rllib.policy import Policy, TorchPolicy
from ray.rllib.policy.policy import LEARNER_STATS_KEY
from ray.rllib.utils.annotations import override

from raylab.algorithms.naf.naf_module import NAFModule


class NAFTorchPolicy(Policy):
    """PyTorch policy to use with RLlib."""

    # pylint: disable=abstract-method

    @override(Policy)
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space)
        self.config = config
        self.device = (
            torch.device("cuda")
            if bool(os.environ.get("CUDA_VISIBLE_DEVICES", None))
            else torch.device("cpu")
        )
        self.module = self._make_module(
            self.observation_space, self.action_space, self.config
        )
        self.optimizer = self._make_optimizer(self.module, self.config)

    @override(Policy)
    def compute_actions(  # pylint: disable=too-many-arguments
        self,
        obs_batch,
        state_batches,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs
    ):
        pass

    @override(Policy)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        return sample_batch

    @override(Policy)
    def learn_on_batch(self, samples):
        info = {}
        return {LEARNER_STATS_KEY: info}

    @override(Policy)
    def get_weights(self):
        return {k: v.cpu() for k, v in self.module.state_dict().items()}

    @override(Policy)
    def set_weights(self, weights):
        self.module.load_state_dict(weights)

    @staticmethod
    def _make_module(obs_space, action_space, config):
        obs_dim = obs_space.shape[0]
        action_dim = action_space.shape[0]
        module = NAFModule(obs_dim, action_dim, config["model"])
        return module

    @staticmethod
    def _make_optimizer(module, config):
        optimizer = config["torch_optimizer"]
        optimizer_options = config["torch_optimizer_options"]
        if isinstance(optimizer, str):
            if optimizer == "Adam":
                optimizer_cls = torch.optim.Adam
            elif optimizer == "RMSprop":
                optimizer_cls = torch.optim.RMSprop
            else:
                raise ValueError("Unsupported optimizer type '{}'.".format(optimizer))
        elif inspect.isclass(optimizer):
            optimizer_cls = optimizer
        else:
            raise ValueError(
                "'torch_optimizer' must be a string or class, got '{}'".format(
                    type(optimizer)
                )
            )

        optimizer = optimizer_cls(module.parameters(), **optimizer_options)
        return optimizer
