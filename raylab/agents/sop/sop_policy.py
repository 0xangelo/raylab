"""SOP policy class using PyTorch."""
import collections

import torch
import torch.nn as nn
from ray.rllib import SampleBatch
from ray.rllib.utils.annotations import override

import raylab.policy as raypi
import raylab.utils.pytorch as ptu
from raylab.losses import ClippedDoubleQLearning


class SOPTorchPolicy(raypi.TargetNetworksMixin, raypi.TorchPolicy):
    """Streamlined Off-Policy policy in PyTorch to use with RLlib."""

    # pylint: disable=abstract-method

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_critic = ClippedDoubleQLearning(
            self.module.critics,
            self.module.target_critics,
            self.module.target_actor,
            gamma=self.config["gamma"],
        )

    @staticmethod
    @override(raypi.TorchPolicy)
    def get_default_config():
        """Return the default configuration for SOP."""
        # pylint: disable=cyclic-import
        from raylab.agents.sop.sop import DEFAULT_CONFIG

        return DEFAULT_CONFIG

    @override(raypi.TorchPolicy)
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

    @override(raypi.TorchPolicy)
    def make_optimizer(self):
        config = self.config["torch_optimizer"]
        components = ["actor", "critics"]

        optims = {k: ptu.build_optimizer(self.module[k], config[k]) for k in components}
        return collections.namedtuple("OptimizerCollection", components)(**optims)

    @override(raypi.TorchPolicy)
    def learn_on_batch(self, samples):
        batch_tensors = self._lazy_tensor_dict(samples)

        info = {}
        info.update(self._update_critic(batch_tensors))
        info.update(self._update_policy(batch_tensors))

        self.update_targets("critics", "target_critics")
        return self._learner_stats(info)

    def _update_critic(self, batch_tensors):
        with self.optimizer.critics.optimize():
            critic_loss, info = self.loss_critic(batch_tensors)
            critic_loss.backward()

        info.update(self.extra_grad_info("critics", batch_tensors))
        return info

    def _update_policy(self, batch_tensors):
        module, config = self.module, self.config
        with self.optimizer.actor.optimize():
            policy_loss, info = self.compute_policy_loss(batch_tensors, module, config)
            policy_loss.backward()

        info.update(self.extra_grad_info("actor", batch_tensors))
        return info

    @staticmethod
    def compute_policy_loss(batch_tensors, module, config):
        """Compute loss for deterministic policy gradient."""
        # pylint: disable=unused-argument
        obs = batch_tensors[SampleBatch.CUR_OBS]

        actions = module.actor(obs)
        action_values, _ = torch.cat(
            [m(obs, actions) for m in module.critics], dim=-1
        ).min(dim=-1)
        max_objective = torch.mean(action_values)

        stats = {
            "policy_loss": max_objective.neg().item(),
            "qpi_mean": max_objective.item(),
        }
        return max_objective.neg(), stats

    @torch.no_grad()
    def extra_grad_info(self, component, batch_tensors):
        """Return statistics right after components are updated."""
        # pylint:disable=unused-argument
        return {
            f"grad_norm({component})": nn.utils.clip_grad_norm_(
                self.module[component].parameters(), float("inf")
            ).item()
        }
