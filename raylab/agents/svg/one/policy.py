"""SVG(1) policy class using PyTorch."""
import warnings

import torch
import torch.nn as nn
from ray.rllib import SampleBatch
from ray.rllib.utils import override

from raylab.agents.svg import SVGTorchPolicy
from raylab.policy import AdaptiveKLCoeffMixin
from raylab.policy import EnvFnMixin
from raylab.policy import TorchPolicy
from raylab.policy.losses import OneStepSVG
from raylab.torch.optim import get_optimizer_class


class SVGOneTorchPolicy(AdaptiveKLCoeffMixin, SVGTorchPolicy):
    """Stochastic Value Gradients policy for off-policy learning."""

    # pylint:disable=abstract-method

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_actor = OneStepSVG(
            lambda s, a, s_: self.module.model.reproduce(s_, self.module.model(s, a)),
            self.module.actor.reproduce,
            self.module.critic,
        )
        self.loss_actor.gamma = self.config["gamma"]

    @property
    @override(SVGTorchPolicy)
    def options(self):
        # pylint:disable=cyclic-import
        from raylab.agents.svg.one import SVGOneTrainer

        return SVGOneTrainer.options

    @override(TorchPolicy)
    def compile(self):
        warnings.warn(f"{type(self).__name__} is incompatible with TorchScript")

    def update_old_policy(self):
        """Copy params of current policy into old one for future KL computation."""
        self.module.old_actor.load_state_dict(self.module.actor.state_dict())

    @override(EnvFnMixin)
    def _set_reward_hook(self):
        self.loss_actor.set_reward_fn(self.reward_fn)

    @override(SVGTorchPolicy)
    def _make_module(self, obs_space, action_space, config):
        config["module"]["replay_kl"] = config["replay_kl"]
        return super()._make_module(obs_space, action_space, config)

    @override(SVGTorchPolicy)
    def _make_optimizers(self):
        """PyTorch optimizer to use."""
        optimizers = super()._make_optimizers()
        cls = get_optimizer_class(self.config["torch_optimizer"]["type"], wrap=True)
        options = self.config["torch_optimizer"]
        modules = {
            "model": self.module.model,
            "actor": self.module.actor,
            "critic": self.module.critic,
        }
        param_groups = [
            dict(params=mod.parameters(), **options[name])
            for name, mod in modules.items()
        ]
        optimizers["all"] = cls(param_groups)
        return optimizers

    @override(SVGTorchPolicy)
    def learn_on_batch(self, samples):
        batch_tensors = self.lazy_tensor_dict(samples)
        batch_tensors, info = self.add_truncated_importance_sampling_ratios(
            batch_tensors
        )

        with self.optimizers.optimize("all"):
            model_value_loss, stats = self.compute_joint_model_value_loss(batch_tensors)
            info.update(stats)
            model_value_loss.backward()

            self.module.model.requires_grad_(False)
            self.module.critic.requires_grad_(False)

            svg_loss, stats = self.loss_actor(batch_tensors)
            info.update(stats)
            kl_loss = self.curr_kl_coeff * self._avg_kl_divergence(batch_tensors)
            (svg_loss + kl_loss).backward()

            self.module.model.requires_grad_(True)
            self.module.critic.requires_grad_(True)

        info.update(self.extra_grad_info(batch_tensors))
        info.update(self.update_kl_coeff(samples))
        self._update_polyak()
        return info

    @torch.no_grad()
    @override(AdaptiveKLCoeffMixin)
    def _kl_divergence(self, sample_batch):
        batch_tensors = self.lazy_tensor_dict(sample_batch)
        return self._avg_kl_divergence(batch_tensors).item()

    def _avg_kl_divergence(self, batch_tensors):
        if self.config["replay_kl"]:
            logp = self.module.actor.log_prob(
                batch_tensors[SampleBatch.CUR_OBS], batch_tensors[SampleBatch.ACTIONS]
            )
            return torch.mean(batch_tensors[SampleBatch.ACTION_LOGP] - logp)

        old_act, old_logp = self.module.old_actor.rsample(
            batch_tensors[SampleBatch.CUR_OBS]
        )
        logp = self.module.actor.log_prob(batch_tensors[SampleBatch.CUR_OBS], old_act)
        return torch.mean(old_logp - logp)

    @torch.no_grad()
    def extra_grad_info(self, batch_tensors):
        """Compute gradient norms and policy statistics."""
        grad_norms = {
            f"grad_norm({k})": nn.utils.clip_grad_norm_(
                self.module[k].parameters(), float("inf")
            )
            for k in "model actor critic".split()
        }
        policy_info = {
            "entropy": self.module.actor.log_prob(
                batch_tensors[SampleBatch.CUR_OBS], batch_tensors[SampleBatch.ACTIONS]
            )
            .mean()
            .neg()
            .item(),
            "curr_kl_coeff": self.curr_kl_coeff,
        }
        return {**grad_norms, **policy_info}
