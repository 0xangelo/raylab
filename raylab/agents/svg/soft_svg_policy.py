"""SoftSVG policy class using PyTorch."""
import collections

import torch
import torch.nn as nn
from ray.rllib import SampleBatch
from ray.rllib.utils.annotations import override

import raylab.utils.pytorch as ptu
from .svg_base_policy import SVGBaseTorchPolicy


class SoftSVGTorchPolicy(SVGBaseTorchPolicy):
    """Stochastic Value Gradients policy for off-policy learning."""

    # pylint: disable=abstract-method

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        if self.config["target_entropy"] == "auto":
            self.config["target_entropy"] = -action_space.shape[0]
        assert "target_critic" in self.module, "SoftSVG needs a target Value function!"

    @staticmethod
    @override(SVGBaseTorchPolicy)
    def get_default_config():
        """Return the default config for SoftSVG"""
        # pylint: disable=cyclic-import
        from raylab.agents.svg.soft_svg import DEFAULT_CONFIG

        return DEFAULT_CONFIG

    @override(SVGBaseTorchPolicy)
    def make_optimizer(self):
        """PyTorch optimizer to use."""
        config = self.config["torch_optimizer"]
        components = "model actor critic alpha".split()

        optims = {k: ptu.build_optimizer(self.module[k], config[k]) for k in components}
        return collections.namedtuple("OptimizerCollection", components)(**optims)

    @override(SVGBaseTorchPolicy)
    def learn_on_batch(self, samples):
        batch_tensors = self._lazy_tensor_dict(samples)
        batch_tensors, info = self.add_importance_sampling_ratios(batch_tensors)

        info.update(self._update_model_and_critic(batch_tensors))
        info.update(self._update_actor(batch_tensors))
        if self.config["target_entropy"] is not None:
            info.update(self._update_alpha(batch_tensors))

        self.update_targets("critic", "target_critic")
        return self._learner_stats(info)

    def _update_model_and_critic(self, batch_tensors):
        with self.optimizer.model.optimize(), self.optimizer.critic.optimize():
            model_value_loss, info = self.compute_joint_model_value_loss(batch_tensors)
            model_value_loss.backward()

        info.update(self.extra_grad_info("model", batch_tensors))
        info.update(self.extra_grad_info("critic", batch_tensors))
        return info

    @override(SVGBaseTorchPolicy)
    def _compute_value_targets(self, batch_tensors):
        _, logp = self.module.actor.sample(batch_tensors[SampleBatch.CUR_OBS])
        rewards = batch_tensors[SampleBatch.REWARDS]
        augmented_rewards = rewards - logp * self.module.alpha()

        next_obs = batch_tensors[SampleBatch.NEXT_OBS]
        next_vals = self.module.target_critic(next_obs).squeeze(-1)

        gamma = self.config["gamma"]
        targets = torch.where(
            batch_tensors[SampleBatch.DONES],
            augmented_rewards,
            augmented_rewards + gamma * next_vals,
        )
        return targets

    def _update_actor(self, batch_tensors):
        with self.optimizer.actor.optimize():
            svg_loss, info = self.compute_stochastic_value_gradient_loss(batch_tensors)
            svg_loss.backward()

        info.update(self.extra_grad_info("actor", batch_tensors))
        return info

    def compute_stochastic_value_gradient_loss(self, batch_tensors):
        """Compute bootstrapped Stochatic Value Gradient loss."""
        is_ratios = batch_tensors[self.IS_RATIOS]
        td_targets = self._compute_policy_td_targets(batch_tensors)
        svg_loss = -torch.mean(is_ratios * td_targets)
        return svg_loss, {"loss(actor)": svg_loss.item()}

    def _compute_policy_td_targets(self, batch_tensors):
        _acts, _logp = self.module.actor.reproduce(
            batch_tensors[SampleBatch.CUR_OBS], batch_tensors[SampleBatch.ACTIONS]
        )
        _next_obs, _ = self.module.model.reproduce(
            batch_tensors[SampleBatch.CUR_OBS],
            _acts,
            batch_tensors[SampleBatch.NEXT_OBS],
        )
        _rewards = self.reward(batch_tensors[SampleBatch.CUR_OBS], _acts, _next_obs)
        _augmented_rewards = _rewards - _logp * self.module.alpha()
        _next_vals = self.module.critic(_next_obs).squeeze(-1)

        gamma = self.config["gamma"]
        return torch.where(
            batch_tensors[SampleBatch.DONES],
            _augmented_rewards,
            _augmented_rewards + gamma * _next_vals,
        )

    def _update_alpha(self, batch_tensors):
        with self.optimizer.alpha.optimize():
            alpha_loss, info = self.compute_alpha_loss(batch_tensors)
            alpha_loss.backward()

        info.update(self.extra_grad_info("actor", batch_tensors))
        return info

    def compute_alpha_loss(self, batch_tensors):
        """Compute entropy coefficient loss."""
        target_entropy = self.config["target_entropy"]

        with torch.no_grad():
            _, logp = self.module.actor.rsample(batch_tensors[SampleBatch.CUR_OBS])

        alpha = self.module.alpha()
        entropy_diff = torch.mean(-alpha * logp - alpha * target_entropy)
        info = {"loss(alpha)": entropy_diff.item(), "curr_alpha": alpha.item()}
        return entropy_diff, info

    @torch.no_grad()
    def extra_grad_info(self, component, batch_tensors):
        """Return gradient statistics for component."""
        fetches = {
            f"grad_norm({component})": nn.utils.clip_grad_norm_(
                self.module[component].parameters(), float("inf")
            ).item()
        }
        if component == "actor":
            _, logp = self.module.actor.sample(batch_tensors[SampleBatch.CUR_OBS])
            fetches["entropy"] = -logp.mean().item()
        return fetches
