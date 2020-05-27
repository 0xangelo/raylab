"""SVG(1) policy class using PyTorch."""
import torch
import torch.nn as nn
from ray.rllib import SampleBatch
from ray.rllib.utils import override

import raylab.utils.pytorch as ptu
from raylab.agents.svg import SVGTorchPolicy
from raylab.losses import OneStepSVG
from raylab.policy import AdaptiveKLCoeffMixin


class SVGOneTorchPolicy(AdaptiveKLCoeffMixin, SVGTorchPolicy):
    """Stochastic Value Gradients policy for off-policy learning."""

    # pylint: disable=abstract-method

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_actor = OneStepSVG(
            self.module.model.reproduce,
            self.module.actor.reproduce,
            self.module.critic,
            self.reward,
            gamma=self.config["gamma"],
        )

    @staticmethod
    @override(SVGTorchPolicy)
    def get_default_config():
        """Return the default config for SVG(1)"""
        # pylint: disable=cyclic-import
        from raylab.agents.svg.one import DEFAULT_CONFIG

        return DEFAULT_CONFIG

    @override(SVGTorchPolicy)
    def make_module(self, obs_space, action_space, config):
        config["module"]["replay_kl"] = config["replay_kl"]
        return super().make_module(obs_space, action_space, config)

    @override(SVGTorchPolicy)
    def make_optimizer(self):
        """PyTorch optimizer to use."""
        optim_cls = ptu.get_optimizer_class(self.config["torch_optimizer"])
        options = self.config["torch_optimizer_options"]
        params = [
            dict(params=self.module[k].parameters(), **options[k]) for k in options
        ]
        return optim_cls(params)

    def update_old_policy(self):
        """Copy params of current policy into old one for future KL computation."""
        self.module.old_actor.load_state_dict(self.module.actor.state_dict())

    @override(SVGTorchPolicy)
    def learn_on_batch(self, samples):
        batch_tensors = self._lazy_tensor_dict(samples)
        batch_tensors, info = self.add_truncated_importance_sampling_ratios(
            batch_tensors
        )

        with self.optimizer.optimize():
            model_value_loss, stats = self.compute_joint_model_value_loss(batch_tensors)
            info.update(stats)
            model_value_loss.backward()

            with self.freeze_nets("model", "critic"):
                svg_loss, stats = self.loss_actor(batch_tensors)
                info.update(stats)
                kl_loss = self.curr_kl_coeff * self._avg_kl_divergence(batch_tensors)
                (svg_loss + kl_loss).backward()

        info.update(self.extra_grad_info(batch_tensors))
        info.update(self.update_kl_coeff(samples))
        self.update_targets("critic", "target_critic")
        return self._learner_stats(info)

    @torch.no_grad()
    @override(AdaptiveKLCoeffMixin)
    def _kl_divergence(self, sample_batch):
        batch_tensors = self._lazy_tensor_dict(sample_batch)
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
