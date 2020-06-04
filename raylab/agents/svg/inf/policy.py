"""SVG(inf) policy class using PyTorch."""
import collections
from contextlib import contextmanager

import torch
import torch.nn as nn
from ray.rllib import SampleBatch
from ray.rllib.utils import override

from raylab.agents.svg import SVGTorchPolicy
from raylab.losses import TrajectorySVG
from raylab.policy import AdaptiveKLCoeffMixin
from raylab.pytorch.optim import build_optimizer


class SVGInfTorchPolicy(AdaptiveKLCoeffMixin, SVGTorchPolicy):
    """Stochastic Value Gradients policy for full trajectories."""

    # pylint: disable=abstract-method

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        # Flag for off-policy learning
        self._off_policy_learning = False

        self.loss_actor = TrajectorySVG(
            self.module.model,
            self.module.actor,
            self.module.critic,
            self.reward,
            gamma=self.config["gamma"],
            torch_script=self.config["module"].get("torch_script", False),
        )

    @staticmethod
    @override(SVGTorchPolicy)
    def get_default_config():
        """Return the default config for SVG(inf)"""
        # pylint: disable=cyclic-import
        from raylab.agents.svg.inf import DEFAULT_CONFIG

        return DEFAULT_CONFIG

    @override(SVGTorchPolicy)
    def make_optimizer(self):
        """PyTorch optimizers to use."""
        config = self.config["torch_optimizer"]
        components = ["on_policy", "off_policy"]
        module = {
            "on_policy": self.module.actor,
            "off_policy": nn.ModuleList([self.module.model, self.module.critic]),
        }

        optims = {k: build_optimizer(module[k], config[k]) for k in components}
        return collections.namedtuple("OptimizerCollection", components)(**optims)

    @override(SVGTorchPolicy)
    def learn_on_batch(self, samples):
        batch_tensors = self._lazy_tensor_dict(samples)
        if self._off_policy_learning:
            info = self._learn_off_policy(batch_tensors)
        else:
            info = self._learn_on_policy(batch_tensors, samples)
        info.update(self.extra_grad_info(batch_tensors))
        return self._learner_stats(info)

    @contextmanager
    def learning_off_policy(self):
        """Signal to policy to use samples for updating off-policy components."""
        old = self._off_policy_learning
        self._off_policy_learning = True
        yield
        self._off_policy_learning = old

    def _learn_off_policy(self, batch_tensors):
        """Update off-policy components."""
        batch_tensors, info = self.add_truncated_importance_sampling_ratios(
            batch_tensors
        )

        with self.optimizer.off_policy.optimize():
            loss, _info = self.compute_joint_model_value_loss(batch_tensors)
            info.update(_info)
            loss.backward()

        self.update_targets("critic", "target_critic")
        return self._learner_stats(info)

    def _learn_on_policy(self, batch_tensors, samples):
        """Update on-policy components."""
        episodes = [self._lazy_tensor_dict(s) for s in samples.split_by_episode()]

        with self.optimizer.on_policy.optimize():
            loss, info = self.loss_actor(episodes)
            kl_div = self._avg_kl_divergence(batch_tensors)
            loss = loss + kl_div * self.curr_kl_coeff
            loss.backward()

        info.update(self.update_kl_coeff(samples))
        return info

    @torch.no_grad()
    @override(AdaptiveKLCoeffMixin)
    def _kl_divergence(self, sample_batch):
        batch_tensors = self._lazy_tensor_dict(sample_batch)
        return self._avg_kl_divergence(batch_tensors).item()

    def _avg_kl_divergence(self, batch_tensors):
        logp = self.module.actor.log_prob(
            batch_tensors[SampleBatch.CUR_OBS], batch_tensors[SampleBatch.ACTIONS]
        )
        return torch.mean(batch_tensors[SampleBatch.ACTION_LOGP] - logp)

    @torch.no_grad()
    def extra_grad_info(self, batch_tensors):
        """Compute gradient norm for components. Also clips on-policy gradient."""
        if self._off_policy_learning:
            model_params = self.module.model.parameters()
            value_params = self.module.critic.parameters()
            fetches = {
                "model_grad_norm": nn.utils.clip_grad_norm_(
                    model_params, float("inf")
                ).item(),
                "value_grad_norm": nn.utils.clip_grad_norm_(
                    value_params, float("inf")
                ).item(),
            }
        else:
            policy_params = self.module.actor.parameters()
            fetches = {
                "policy_grad_norm": nn.utils.clip_grad_norm_(
                    policy_params, max_norm=self.config["max_grad_norm"]
                ).item(),
                "policy_entropy": -batch_tensors[SampleBatch.ACTION_LOGP].mean().item(),
                "curr_kl_coeff": self.curr_kl_coeff,
            }
        return fetches
