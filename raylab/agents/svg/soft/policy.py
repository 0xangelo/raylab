"""SoftSVG policy class using PyTorch."""
import torch
import torch.nn as nn
from ray.rllib import SampleBatch
from ray.rllib.utils import override

from raylab.agents.svg import SVGTorchPolicy
from raylab.policy import EnvFnMixin
from raylab.policy.losses import ISSoftVIteration
from raylab.policy.losses import MaximumEntropyDual
from raylab.policy.losses import OneStepSoftSVG
from raylab.torch.optim import build_optimizer


class SoftSVGTorchPolicy(SVGTorchPolicy):
    """Stochastic Value Gradients policy for off-policy learning."""

    # pylint:disable=abstract-method

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        assert "target_critic" in self.module, "SoftSVG needs a target Value function!"

        self.loss_actor = OneStepSoftSVG(
            lambda s, a, s_: self.module.model.reproduce(s_, self.module.model(s, a)),
            self.module.actor.reproduce,
            self.module.critic,
        )
        self.loss_actor.gamma = self.config["gamma"]

        self.loss_critic = ISSoftVIteration(
            self.module.critic, self.module.target_critic, self.module.actor.sample,
        )
        self.loss_critic.gamma = self.config["gamma"]

        target_entropy = (
            -action_space.shape[0]
            if self.config["target_entropy"] == "auto"
            else self.config["target_entropy"]
        )
        self.loss_alpha = MaximumEntropyDual(
            self.module.alpha, self.module.actor.sample, target_entropy
        )

    @property
    @override(SVGTorchPolicy)
    def options(self):
        # pylint:disable=cyclic-import
        from raylab.agents.svg.soft import SoftSVGTrainer

        return SoftSVGTrainer.options

    @override(EnvFnMixin)
    def _set_reward_hook(self):
        self.loss_actor.set_reward_fn(self.reward_fn)

    @override(SVGTorchPolicy)
    def _make_optimizers(self):
        optimizers = super()._make_optimizers()
        config = self.config["torch_optimizer"]
        components = {
            "model": self.module.model,
            "actor": self.module.actor,
            "critic": self.module.critic,
            "alpha": self.module.alpha,
        }

        mapping = {
            name: build_optimizer(module, config[name])
            for name, module in components.items()
        }

        optimizers.update(mapping)
        return optimizers

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
        batch_tensors = self.lazy_tensor_dict(samples)
        batch_tensors, info = self.add_truncated_importance_sampling_ratios(
            batch_tensors
        )

        alpha = self.module.alpha().item()
        self.loss_critic.alpha = alpha
        self.loss_actor.alpha = alpha

        info.update(self._update_model(batch_tensors))
        info.update(self._update_critic(batch_tensors))
        info.update(self._update_actor(batch_tensors))
        if self.config["target_entropy"] is not None:
            info.update(self._update_alpha(batch_tensors))

        self._update_polyak()
        return info

    def _update_model(self, batch_tensors):
        with self.optimizers.optimize("model"):
            model_loss, info = self.loss_model(batch_tensors)
            model_loss.backward()

        info.update(self.extra_grad_info("model"))
        return info

    def _update_critic(self, batch_tensors):
        with self.optimizers.optimize("critic"):
            value_loss, info = self.loss_critic(batch_tensors)
            value_loss.backward()

        info.update(self.extra_grad_info("critic"))
        return info

    def _update_actor(self, batch_tensors):
        with self.optimizers.optimize("actor"):
            svg_loss, info = self.loss_actor(batch_tensors)
            svg_loss.backward()

        info.update(self.extra_grad_info("actor"))
        return info

    def _update_alpha(self, batch_tensors):
        with self.optimizers.optimize("alpha"):
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
