"""SVG(inf) policy class using PyTorch."""
from functools import partialmethod

import torch
import torch.nn as nn
from ray.rllib.policy.policy import LEARNER_STATS_KEY
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override

from raylab.algorithms.svg.svg_base_policy import SVGBaseTorchPolicy
import raylab.modules as modules
import raylab.utils.pytorch as torch_util


class SVGOneTorchPolicy(SVGBaseTorchPolicy):
    """Stochastic Value Gradients policy for off-policy learning."""

    # pylint: disable=abstract-method

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self._optimizer = self.optimizer()

    @staticmethod
    @override(SVGBaseTorchPolicy)
    def get_default_config():
        """Return the default config for SVG(1)"""
        # pylint: disable=cyclic-import
        from raylab.algorithms.svg.svg_one import DEFAULT_CONFIG

        return DEFAULT_CONFIG

    @override(SVGBaseTorchPolicy)
    def learn_on_batch(self, samples):
        batch_tensors = self._lazy_tensor_dict(samples)
        batch_tensors, info = self.add_importance_sampling_ratios(batch_tensors)
        self._optimizer.zero_grad()

        self._unfreeze_non_policy_grads()
        model_value_loss, _info = self.compute_joint_model_value_loss(batch_tensors)
        info.update(_info)
        model_value_loss.backward()

        self._freeze_non_policy_grads()
        pi_loss, _info = self.compute_stochastic_value_gradient_loss(batch_tensors)
        info.update(_info)
        pi_loss = pi_loss + self.curr_kl_coeff * self._avg_kl_divergence(batch_tensors)
        pi_loss.backward()

        info.update(self.extra_grad_info(batch_tensors))
        self._optimizer.step()
        self.update_targets()

        return {LEARNER_STATS_KEY: info}

    @override(SVGBaseTorchPolicy)
    def optimizer(self):
        """PyTorch optimizer to use."""
        optim_cls = torch_util.get_optimizer_class(self.config["torch_optimizer"])
        options = self.config["torch_optimizer_options"]
        params = [
            dict(params=self.module[k].parameters(), **options[k]) for k in options
        ]
        return optim_cls(params)

    @override(SVGBaseTorchPolicy)
    def set_reward_fn(self, reward_fn):
        self.module.reward = modules.Lambda(reward_fn)

    # ================================= NEW METHODS ====================================

    def compute_stochastic_value_gradient_loss(self, batch_tensors):
        """Compute bootstrapped Stochatic Value Gradient loss."""
        td_targets = self._compute_policy_td_targets(batch_tensors)
        svg_loss = torch.mean(batch_tensors[self.IS_RATIOS] * td_targets).neg()
        return svg_loss, {"svg_loss": svg_loss.item()}

    @torch.no_grad()
    def extra_grad_info(self, batch_tensors):
        """Compute gradient norm for components. Also clips policy gradient."""
        model_params = self.module.model.parameters()
        value_params = self.module.value.parameters()
        policy_params = self.module.policy.parameters()
        fetches = {
            "model_grad_norm": nn.utils.clip_grad_norm_(model_params, float("inf")),
            "value_grad_norm": nn.utils.clip_grad_norm_(value_params, float("inf")),
            "policy_grad_norm": nn.utils.clip_grad_norm_(
                policy_params, max_norm=self.config["max_grad_norm"]
            ),
            "policy_entropy": self.module.entropy(
                self.module.policy(batch_tensors[SampleBatch.CUR_OBS])
            )
            .mean()
            .item(),
            "curr_kl_coeff": self.curr_kl_coeff,
        }
        return fetches

    def _set_non_policy_require_grad(self, require_grad):
        for name in ("model", "value"):
            self.module[name].requires_grad_(require_grad)

    _freeze_non_policy_grads = partialmethod(_set_non_policy_require_grad, False)
    _unfreeze_non_policy_grads = partialmethod(_set_non_policy_require_grad, True)

    def _compute_policy_td_targets(self, batch_tensors):
        _acts = self.module.policy_rsample(
            self.module.policy(batch_tensors[SampleBatch.CUR_OBS]),
            batch_tensors[SampleBatch.ACTIONS],
        )
        _residual = self.module.model_rsample(
            self.module.model(batch_tensors[SampleBatch.CUR_OBS], _acts),
            batch_tensors[SampleBatch.NEXT_OBS] - batch_tensors[SampleBatch.CUR_OBS],
        )
        _next_obs = batch_tensors[SampleBatch.CUR_OBS] + _residual
        _rewards = self.module.reward(
            batch_tensors[SampleBatch.CUR_OBS], _acts, _next_obs
        )
        _next_vals = self.module.value(_next_obs).squeeze(-1)
        td_targets = torch.where(
            batch_tensors[SampleBatch.DONES],
            _rewards,
            _rewards + self.config["gamma"] * _next_vals,
        )
        return td_targets
