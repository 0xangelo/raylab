"""NAF policy class using PyTorch."""
import os
import inspect

import torch
import torch.nn as nn
from ray.rllib.utils import merge_dicts
from ray.rllib.policy import Policy, TorchPolicy
from ray.rllib.policy.policy import LEARNER_STATS_KEY
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override

from raylab.utils.pytorch import convert_to_tensor, get_optimizer_class, update_polyak
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

        # Flag for uniform random actions
        self._pure_exploration = False

    @override(Policy)
    @torch.no_grad()
    def compute_actions(
        self,
        obs_batch,
        state_batches,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs
    ):
        # pylint: disable=too-many-arguments,unused-argument
        obs_batch = convert_to_tensor(obs_batch, self.device)

        if self._pure_exploration:
            actions = self._uniform_random_actions(obs_batch)
        elif self.config["exploration"] == "full_gaussian":
            actions = self._multivariate_gaussian_actions(obs_batch)
        else:
            actions = self._greedy_actions(obs_batch, self.module["main"])

        return actions.cpu().numpy(), state_batches, {}

    @override(Policy)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        horizon = self.config["horizon"]
        if (
            episode
            and horizon
            and self.config["timeout_bootstrap"]
            and episode.length >= horizon
        ):
            sample_batch[SampleBatch.DONES][-1] = False

        return sample_batch

    @override(Policy)
    def learn_on_batch(self, samples):
        # pylint: disable=protected-access
        batch_tensors = TorchPolicy._lazy_tensor_dict(self, samples)
        loss, info = self.compute_loss(batch_tensors, self.module, self.config)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        update_polyak(self.module["main"], self.module["target"], self.config["polyak"])
        return {LEARNER_STATS_KEY: info}

    @override(Policy)
    def get_weights(self):
        return {k: v.cpu() for k, v in self.module.state_dict().items()}

    @override(Policy)
    def set_weights(self, weights):
        self.module.load_state_dict(weights)

    # === New Methods ===

    # === Exploration ===
    def set_pure_exploration_phase(self, phase):
        """Set a boolean flag that tells the policy to act randomly."""
        self._pure_exploration = phase

    # === Action Sampling ===
    def _uniform_random_actions(self, obs_batch):
        dist = torch.distributions.Uniform(
            convert_to_tensor(self.action_space.low, self.device),
            convert_to_tensor(self.action_space.high, self.device),
        )
        actions = dist.sample(sample_shape=obs_batch.shape[:-1])
        return actions

    @staticmethod
    def _greedy_actions(obs_batch, module):
        logits = module.logits_module(obs_batch)
        actions = module.action_module(logits)
        return actions

    def _multivariate_gaussian_actions(self, obs_batch):
        module = self.module["main"]
        logits = module.logits_module(obs_batch)
        loc = module.action_module(logits)
        scale_tril = module.advantage_module.tril_matrix_module(logits)
        scale_coeff = self.config["scale_tril_coeff"]
        dist = torch.distributions.MultivariateNormal(
            loc=loc, scale_tril=scale_tril * scale_coeff
        )
        actions = dist.sample()
        return actions

    # === Static Methods ===

    @staticmethod
    def compute_loss(batch_tensors, module, config):
        """Compute the forward pass of NAF's loss function.

        Arguments:
            batch_tensors (UsageTrackingDict): Dictionary of experience batches that are
                lazily converted to tensors.
            module (nn.Module): The module of the policy
            config (dict): The policy's configuration

        Returns:
            A scalar tensor sumarizing the losses for this experience batch.
        """
        gamma = config["gamma"]
        obs = batch_tensors[SampleBatch.CUR_OBS]
        actions = batch_tensors[SampleBatch.ACTIONS]
        rewards = batch_tensors[SampleBatch.REWARDS]
        dones = batch_tensors[SampleBatch.DONES]
        next_obs = batch_tensors[SampleBatch.NEXT_OBS]

        next_logits = module["target"].logits_module(next_obs)
        best_next_value = module["target"].value_module(next_logits)
        best_next_value.squeeze_(-1)
        target_value = torch.where(dones, rewards, rewards + gamma * best_next_value)
        action_value, _, _ = module["main"](obs, actions)
        action_value.squeeze_(-1)
        return torch.nn.MSELoss()(action_value, target_value), {}

    @staticmethod
    def get_default_config():
        from raylab.algorithms.naf.naf import DEFAULT_CONFIG

        return DEFAULT_CONFIG

    @staticmethod
    def _make_module(obs_space, action_space, config):
        obs_dim = obs_space.shape[0]
        action_low = torch.from_numpy(action_space.low).float()
        action_high = torch.from_numpy(action_space.high).float()
        module = nn.ModuleDict()
        module["main"] = NAFModule(obs_dim, action_low, action_high, config["module"])
        module["target"] = NAFModule(obs_dim, action_low, action_high, config["module"])
        module["target"].load_state_dict(module["main"].state_dict())
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
