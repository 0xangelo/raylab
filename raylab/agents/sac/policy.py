"""SAC policy class using PyTorch."""
import collections

import torch
import torch.nn as nn
from ray.rllib.utils import override

from raylab.losses import MaximumEntropyDual
from raylab.losses import ReparameterizedSoftPG
from raylab.losses import SoftCDQLearning
from raylab.policy import TargetNetworksMixin
from raylab.policy import TorchPolicy
from raylab.pytorch.optim import build_optimizer


class SACTorchPolicy(TargetNetworksMixin, TorchPolicy):
    """Soft Actor-Critic policy in PyTorch to use with RLlib."""

    # pylint: disable=abstract-method

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self.loss_actor = ReparameterizedSoftPG(
            self.module.actor, self.module.critics, self.module.alpha
        )
        self.loss_critic = SoftCDQLearning(
            self.module.critics,
            self.module.target_critics,
            self.module.actor,
            gamma=self.config["gamma"],
            alpha=self.module.alpha,
        )
        target_entropy = (
            -action_space.shape[0]
            if self.config["target_entropy"] == "auto"
            else self.config["target_entropy"]
        )
        self.loss_alpha = MaximumEntropyDual(
            self.module.alpha, self.module.actor.sample, target_entropy
        )

    @staticmethod
    @override(TorchPolicy)
    def get_default_config():
        """Return the default config for SAC."""
        # pylint: disable=cyclic-import
        from raylab.agents.sac import DEFAULT_CONFIG

        return DEFAULT_CONFIG

    @override(TorchPolicy)
    def make_module(self, obs_space, action_space, config):
        module_config = config["module"]
        module_config.setdefault("critic", {})
        module_config["critic"]["double_q"] = config["clipped_double_q"]
        return super().make_module(obs_space, action_space, config)

    @override(TorchPolicy)
    def make_optimizer(self):
        config = self.config["torch_optimizer"]
        components = "actor critics alpha".split()

        optims = {k: build_optimizer(self.module[k], config[k]) for k in components}
        return collections.namedtuple("OptimizerCollection", components)(**optims)

    @override(TorchPolicy)
    def learn_on_batch(self, samples):
        batch_tensors = self._lazy_tensor_dict(samples)
        info = {}

        info.update(self._update_critic(batch_tensors))
        info.update(self._update_actor(batch_tensors))
        if self.config["target_entropy"] is not None:
            info.update(self._update_alpha(batch_tensors))

        self.update_targets("critics", "target_critics")
        return self._learner_stats(info)

    def _update_critic(self, batch_tensors):
        with self.optimizer.critics.optimize():
            critic_loss, info = self.loss_critic(batch_tensors)
            critic_loss.backward()

        info.update(self.extra_grad_info("critics"))
        return info

    def _update_actor(self, batch_tensors):
        with self.optimizer.actor.optimize():
            actor_loss, info = self.loss_actor(batch_tensors)
            actor_loss.backward()

        info.update(self.extra_grad_info("actor"))
        return info

    def _update_alpha(self, batch_tensors):
        with self.optimizer.alpha.optimize():
            alpha_loss, info = self.loss_alpha(batch_tensors)
            alpha_loss.backward()

        info.update(self.extra_grad_info("alpha"))
        return info

    @torch.no_grad()
    def extra_grad_info(self, component):
        """Return statistics right after components are updated."""
        return {
            f"grad_norm({component})": nn.utils.clip_grad_norm_(
                self.module[component].parameters(), float("inf")
            ).item()
        }
