"""NAF policy class using PyTorch."""
import os
import inspect

import torch
from ray.rllib.utils import merge_dicts
from ray.rllib.policy.policy import Policy, LEARNER_STATS_KEY
from ray.rllib.utils.annotations import override

from raylab.utils.pytorch import convert_to_tensor, get_optimizer_class
from raylab.algorithms.naf.naf_module import NAFModule


class NAFTorchPolicy(Policy):
    """Normalized Advantage Function policy in Pytorch to use with RLlib."""

    # pylint: disable=abstract-method

    @override(Policy)
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self.config = merge_dicts(self.get_default_config(), config)
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
    def compute_actions(  # pylint: disable=too-many-arguments,unused-argument
        self,
        obs_batch,
        state_batches,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs
    ):
        obs = convert_to_tensor(obs_batch, self.device)
        with torch.no_grad():
            logits = self.module.logits_module(obs)
            action = self.module.action_module(logits)
        return action.cpu().numpy(), state_batches, {}

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
    def get_default_config():
        from raylab.algorithms.naf.naf import DEFAULT_CONFIG

        return DEFAULT_CONFIG

    @staticmethod
    def _make_module(obs_space, action_space, config):
        obs_dim = obs_space.shape[0]
        action_low = torch.from_numpy(action_space.low).float()
        action_high = torch.from_numpy(action_space.high).float()
        module = NAFModule(obs_dim, action_low, action_high, config["model"])
        return module

    @staticmethod
    def _make_optimizer(module, config):
        optimizer = config["torch_optimizer"]
        if isinstance(optimizer, str):
            optimizer_cls = get_optimizer_class(optimizer)
        elif inspect.isclass(optimizer):
            optimizer_cls = optimizer
        else:
            raise ValueError(
                "'torch_optimizer' must be a string or class, got '{}'".format(
                    type(optimizer)
                )
            )

        optimizer_options = config["torch_optimizer_options"]
        optimizer = optimizer_cls(module.parameters(), **optimizer_options)
        return optimizer
