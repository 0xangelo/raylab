"""SOP policy class using PyTorch."""
import torch
import torch.nn as nn
from ray.rllib.utils import override

from raylab.losses import ClippedDoubleQLearning
from raylab.losses import DeterministicPolicyGradient
from raylab.policy import TargetNetworksMixin
from raylab.policy import TorchPolicy
from raylab.pytorch.optim import build_optimizer


class SOPTorchPolicy(TargetNetworksMixin, TorchPolicy):
    """Streamlined Off-Policy policy in PyTorch to use with RLlib."""

    # pylint: disable=abstract-method

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_actor = DeterministicPolicyGradient(
            self.module.actor, self.module.critics,
        )
        self.loss_critic = ClippedDoubleQLearning(
            self.module.critics, self.module.target_critics, self.module.target_actor,
        )
        self.loss_critic.gamma = self.config["gamma"]
        self._grad_step = 0

    @staticmethod
    @override(TorchPolicy)
    def get_default_config():
        """Return the default configuration for SOP."""
        # pylint: disable=cyclic-import
        from raylab.agents.sop import DEFAULT_CONFIG

        return DEFAULT_CONFIG

    @override(TorchPolicy)
    def make_module(self, obs_space, action_space, config):
        module_config = config["module"]
        module_config.setdefault("critic", {})
        module_config["critic"]["double_q"] = config["clipped_double_q"]
        module_config.setdefault("actor", {})
        module_config["actor"]["perturbed_policy"] = (
            config["exploration_config"]["type"]
            == "raylab.utils.exploration.ParameterNoise"
        )
        # pylint:disable=no-member
        return super().make_module(obs_space, action_space, config)

    @override(TorchPolicy)
    def make_optimizers(self):
        config = self.config["torch_optimizer"]
        components = "actor critics".split()

        return {
            name: build_optimizer(self.module[name], config[name])
            for name in components
        }

    @override(TorchPolicy)
    def learn_on_batch(self, samples):
        batch_tensors = self.lazy_tensor_dict(samples)

        info = {}
        self._grad_step += 1
        info.update(self._update_critic(batch_tensors))
        if self._grad_step % self.config["policy_delay"] == 0:
            info.update(self._update_policy(batch_tensors))

        self.update_targets("critics", "target_critics")
        return info

    def _update_critic(self, batch_tensors):
        with self.optimizers.optimize("critics"):
            loss, info = self.loss_critic(batch_tensors)
            loss.backward()

        info.update(self.extra_grad_info("critics"))
        return info

    def _update_policy(self, batch_tensors):
        with self.optimizers.optimize("actor"):
            loss, info = self.loss_actor(batch_tensors)
            loss.backward()

        info.update(self.extra_grad_info("actor"))
        return info

    @torch.no_grad()
    def extra_grad_info(self, component):
        """Return statistics right after components are updated."""
        return {
            f"grad_norm({component})": nn.utils.clip_grad_norm_(
                self.module[component].parameters(), float("inf")
            ).item()
        }

    @override(TorchPolicy)
    def get_weights(self):
        weights = super().get_weights()
        weights["grad_step"] = self._grad_step
        return weights

    @override(TorchPolicy)
    def set_weights(self, weights):
        self._grad_step = weights["grad_step"]
        super().set_weights(weights)
