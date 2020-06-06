"""SoftSVG policy class using PyTorch."""
import collections

import torch
import torch.nn as nn
from ray.rllib import SampleBatch
from ray.rllib.utils import override

from raylab.agents.svg import SVGTorchPolicy
from raylab.losses import ISSoftVIteration
from raylab.losses import MaximumEntropyDual
from raylab.losses import OneStepSoftSVG
from raylab.policy import EnvFnMixin
from raylab.pytorch.optim import build_optimizer


class SoftSVGTorchPolicy(SVGTorchPolicy):
    """Stochastic Value Gradients policy for off-policy learning."""

    # pylint: disable=abstract-method

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        assert "target_critic" in self.module, "SoftSVG needs a target Value function!"

        self.loss_actor = OneStepSoftSVG(
            self.module.model.reproduce,
            self.module.actor.reproduce,
            self.module.critic,
            alpha=self.module.alpha,
            gamma=self.config["gamma"],
        )
        self.loss_critic = ISSoftVIteration(
            self.module.critic,
            self.module.target_critic,
            self.module.actor.sample,
            self.module.alpha,
            gamma=self.config["gamma"],
        )
        target_entropy = (
            -action_space.shape[0]
            if self.config["target_entropy"] == "auto"
            else self.config["target_entropy"]
        )
        self.loss_alpha = MaximumEntropyDual(
            self.module.alpha, self.module.actor.sample, target_entropy
        )

    @override(EnvFnMixin)
    def set_reward_from_config(self, env_name: str, env_config: dict):
        super().set_reward_from_config(env_name, env_config)
        self.loss_actor.set_reward_fn(self.reward_fn)

    @staticmethod
    @override(SVGTorchPolicy)
    def get_default_config():
        """Return the default config for SoftSVG"""
        # pylint: disable=cyclic-import
        from raylab.agents.svg.soft import DEFAULT_CONFIG

        return DEFAULT_CONFIG

    @override(SVGTorchPolicy)
    def make_optimizer(self):
        """PyTorch optimizer to use."""
        config = self.config["torch_optimizer"]
        components = "model actor critic alpha".split()

        optims = {k: build_optimizer(self.module[k], config[k]) for k in components}
        return collections.namedtuple("OptimizerCollection", components)(**optims)

    @torch.no_grad()
    @override(SVGTorchPolicy)
    def add_truncated_importance_sampling_ratios(self, batch_tensors):
        """Compute and add truncated importance sampling ratios to tensor batch."""
        curr_logp = self.module.actor.log_prob(
            batch_tensors[SampleBatch.CUR_OBS], batch_tensors[SampleBatch.ACTIONS]
        )

        is_ratios = torch.exp(curr_logp - batch_tensors[SampleBatch.ACTION_LOGP])
        _is_ratios = torch.clamp(is_ratios, max=self.config["max_is_ratio"])

        batch_tensors[self.loss_actor.IS_RATIOS] = _is_ratios
        batch_tensors[self.loss_critic.IS_RATIOS] = _is_ratios

        info = {
            "is_ratio_max": is_ratios.max().item(),
            "is_ratio_mean": is_ratios.mean().item(),
            "is_ratio_min": is_ratios.min().item(),
            "cross_entropy": -curr_logp.mean().item(),
        }
        return batch_tensors, info

    @override(SVGTorchPolicy)
    def learn_on_batch(self, samples):
        batch_tensors = self._lazy_tensor_dict(samples)
        batch_tensors, info = self.add_truncated_importance_sampling_ratios(
            batch_tensors
        )

        info.update(self._update_model(batch_tensors))
        info.update(self._update_critic(batch_tensors))
        info.update(self._update_actor(batch_tensors))
        if self.config["target_entropy"] is not None:
            info.update(self._update_alpha(batch_tensors))

        self.update_targets("critic", "target_critic")
        return self._learner_stats(info)

    def _update_model(self, batch_tensors):
        with self.optimizer.model.optimize():
            model_loss, info = self.loss_model(batch_tensors)
            model_loss.backward()

        info.update(self.extra_grad_info("model"))
        return info

    def _update_critic(self, batch_tensors):
        with self.optimizer.critic.optimize():
            value_loss, info = self.loss_critic(batch_tensors)
            value_loss.backward()

        info.update(self.extra_grad_info("critic"))
        return info

    def _update_actor(self, batch_tensors):
        with self.optimizer.actor.optimize():
            svg_loss, info = self.loss_actor(batch_tensors)
            svg_loss.backward()

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
        """Return gradient statistics for component."""
        fetches = {
            f"grad_norm({component})": nn.utils.clip_grad_norm_(
                self.module[component].parameters(), float("inf")
            ).item()
        }
        return fetches
