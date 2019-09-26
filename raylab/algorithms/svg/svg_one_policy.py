"""SVG(inf) policy class using PyTorch."""
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
        loss, info = self.compute_joint_loss(batch_tensors)
        loss = loss + self.curr_kl_coeff * self._avg_kl_divergence(batch_tensors)
        self._optimizer.zero_grad()
        loss.backward()
        info.update(self.extra_grad_info(batch_tensors))
        self._optimizer.step()
        info.update(self.extra_apply_info(batch_tensors))
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

    def compute_joint_loss(self, batch_tensors):  # pylint: disable=too-many-locals
        """Compute model MLE, fitted V, and policy losses."""
        mle_loss = self._avg_model_logp(batch_tensors).neg()

        with torch.no_grad():
            targets = self._compute_value_targets(batch_tensors)
            is_ratio = self._compute_is_ratios(batch_tensors)
            _is_ratio = torch.clamp(is_ratio, max=self.config["max_is_ratio"])

        values = self.module.value(batch_tensors[SampleBatch.CUR_OBS]).squeeze(-1)
        value_loss = torch.mean(
            _is_ratio * nn.MSELoss(reduction="none")(values, targets) / 2
        )

        td_targets = self._compute_policy_td_targets(batch_tensors)
        svg_loss = torch.mean(_is_ratio * td_targets).neg()

        loss = mle_loss + self.config["vf_loss_coeff"] * value_loss + svg_loss
        info = {
            "mle_loss": mle_loss.item(),
            "value_loss": value_loss.item(),
            "svg_loss": svg_loss.item(),
            "avg_is_ratio": is_ratio.mean().item(),
        }
        return loss, info

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
        }
        return fetches

    @torch.no_grad()
    def extra_apply_info(self, batch_tensors):
        """Add average KL divergence between new and old policies."""
        return {"policy_kl_div": self._avg_kl_divergence(batch_tensors).item()}

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
