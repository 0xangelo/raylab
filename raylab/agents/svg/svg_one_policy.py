"""SVG(1) policy class using PyTorch."""
import torch
import torch.nn as nn
from ray.rllib import SampleBatch
from ray.rllib.utils.annotations import override

from raylab.policy import AdaptiveKLCoeffMixin
import raylab.utils.pytorch as ptu
from .svg_base_policy import SVGBaseTorchPolicy


class SVGOneTorchPolicy(AdaptiveKLCoeffMixin, SVGBaseTorchPolicy):
    """Stochastic Value Gradients policy for off-policy learning."""

    # pylint: disable=abstract-method

    @staticmethod
    @override(SVGBaseTorchPolicy)
    def get_default_config():
        """Return the default config for SVG(1)"""
        # pylint: disable=cyclic-import
        from raylab.agents.svg.svg_one import DEFAULT_CONFIG

        return DEFAULT_CONFIG

    @override(SVGBaseTorchPolicy)
    def make_module(self, obs_space, action_space, config):
        config["module"]["replay_kl"] = config["replay_kl"]
        return super().make_module(obs_space, action_space, config)

    @override(SVGBaseTorchPolicy)
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

    @override(SVGBaseTorchPolicy)
    def learn_on_batch(self, samples):
        batch_tensors = self._lazy_tensor_dict(samples)
        batch_tensors, info = self.add_importance_sampling_ratios(batch_tensors)

        with self.optimizer.optimize():
            model_value_loss, stats = self.compute_joint_model_value_loss(batch_tensors)
            info.update(stats)
            model_value_loss.backward()

            with self.freeze_nets("model", "critic"):
                svg_loss, stats = self.compute_stochastic_value_gradient_loss(
                    batch_tensors
                )
                info.update(stats)
                kl_loss = self.curr_kl_coeff * self._avg_kl_divergence(batch_tensors)
                (svg_loss + kl_loss).backward()

        info.update(self.extra_grad_info(batch_tensors))
        info.update(self.update_kl_coeff(samples))
        self.update_targets("critic", "target_critic")
        return self._learner_stats(info)

    def compute_stochastic_value_gradient_loss(self, batch_tensors):
        """Compute bootstrapped Stochatic Value Gradient loss."""
        td_targets = self._compute_policy_td_targets(batch_tensors)
        svg_loss = torch.mean(batch_tensors[self.IS_RATIOS] * td_targets).neg()
        return svg_loss, {"svg_loss": svg_loss.item()}

    def _compute_policy_td_targets(self, batch_tensors):
        _acts, _ = self.module.actor.reproduce(
            batch_tensors[SampleBatch.CUR_OBS], batch_tensors[SampleBatch.ACTIONS]
        )
        _next_obs, _ = self.module.model.reproduce(
            batch_tensors[SampleBatch.CUR_OBS],
            _acts,
            batch_tensors[SampleBatch.NEXT_OBS],
        )
        _rewards = self.reward(batch_tensors[SampleBatch.CUR_OBS], _acts, _next_obs)
        _next_vals = self.module.critic(_next_obs).squeeze(-1)
        return torch.where(
            batch_tensors[SampleBatch.DONES],
            _rewards,
            _rewards + self.config["gamma"] * _next_vals,
        )

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
        """Compute gradient norm for components. Also clips policy gradient."""
        model_params = self.module.model.parameters()
        value_params = self.module.critic.parameters()
        policy_params = self.module.actor.parameters()
        fetches = {
            "model_grad_norm": nn.utils.clip_grad_norm_(
                model_params, float("inf")
            ).item(),
            "value_grad_norm": nn.utils.clip_grad_norm_(
                value_params, float("inf")
            ).item(),
            "policy_grad_norm": nn.utils.clip_grad_norm_(
                policy_params, max_norm=self.config["max_grad_norm"]
            ).item(),
            "policy_entropy": self.module.actor.log_prob(
                batch_tensors[SampleBatch.CUR_OBS], batch_tensors[SampleBatch.ACTIONS]
            )
            .mean()
            .neg()
            .item(),
            "curr_kl_coeff": self.curr_kl_coeff,
        }
        return fetches
