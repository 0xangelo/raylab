"""SVG(inf) policy class using PyTorch."""
import torch
import torch.nn as nn
from ray.rllib.policy.policy import LEARNER_STATS_KEY
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override

from raylab.policy import TorchPolicy
from raylab.algorithms.svg.svg_inf_policy import SVGInfTorchPolicy, Transition
import raylab.modules as modules
import raylab.utils.pytorch as torch_util


class SVGOneTorchPolicy(SVGInfTorchPolicy):
    """Stochastic Value Gradients policy for off-policy learning."""

    # pylint: disable=abstract-method

    def __init__(self, observation_space, action_space, config):
        # pylint: disable=non-parent-init-called,super-init-not-called
        TorchPolicy.__init__(self, observation_space, action_space, config)

        self.module = self._make_module(
            self.observation_space, self.action_space, self.config
        )
        self._optimizer = self.optimizer()

    @staticmethod
    @override(SVGInfTorchPolicy)
    def get_default_config():
        """Return the default config for SVG(1)"""
        # pylint: disable=cyclic-import
        from raylab.algorithms.svg.svg_one import DEFAULT_CONFIG

        return DEFAULT_CONFIG

    @override(SVGInfTorchPolicy)
    def learn_on_batch(self, samples):
        batch_tensors = self._lazy_tensor_dict(samples)
        loss, info = self.compute_joint_loss(batch_tensors)
        self._optimizer.zero_grad()
        loss.backward()
        info.update(self.extra_grad_info())
        self._optimizer.step()
        self.update_targets()

        return {LEARNER_STATS_KEY: info}

    @override(SVGInfTorchPolicy)
    def optimizer(self):
        """PyTorch optimizer to use."""
        optim_cls = torch_util.get_optimizer_class(self.config["torch_optimizer"])
        options = self.config["torch_optimizer_options"]
        params = [
            dict(params=self.module[k].parameters(), **options[k]) for k in options
        ]
        return optim_cls(params)

    @override(SVGInfTorchPolicy)
    def set_reward_fn(self, reward_fn):
        """Set the reward function to use when unrolling the policy and model."""
        # Add recurrent policy-model combination
        self.module.reward = modules.Lambda(reward_fn)

    def compute_joint_loss(self, batch_tensors):  # pylint: disable=too-many-locals
        """Compute model MLE, fitted V, and policy losses."""
        trans = [
            batch_tensors[c]
            for c in (
                SampleBatch.CUR_OBS,
                SampleBatch.ACTIONS,
                SampleBatch.REWARDS,
                SampleBatch.NEXT_OBS,
                SampleBatch.DONES,
            )
        ]
        trans = Transition(*trans)

        dist_params = self.module.model(trans.obs, trans.actions)
        residual = trans.next_obs - trans.obs
        mle_loss = self.module.model_logp(dist_params, residual).mean().neg()

        with torch.no_grad():
            next_vals = self.module.target_value(trans.next_obs).squeeze(-1)
            curr_logp = self.module.policy_logp(
                self.module.policy(trans.obs), trans.actions
            )
            is_ratio = torch.clamp(
                torch.exp(curr_logp - batch_tensors[self.ACTION_LOGP]),
                max=self.config["max_is_ratio"],
            )

        targets = torch.where(
            trans.dones, trans.rewards, trans.rewards + self.config["gamma"] * next_vals
        )
        values = self.module.value(trans.obs).squeeze(-1)
        value_loss = torch.mean(
            is_ratio * nn.MSELoss(reduction="none")(values, targets) / 2
        )

        _acts = self.module.policy_rsample(self.module.policy(trans.obs), trans.actions)
        residual = self.module.model_rsample(
            self.module.model(trans.obs, _acts), trans.next_obs - trans.obs
        )
        _next_obs = trans.obs + residual
        svg_loss = torch.mean(
            is_ratio
            * (
                self.module.reward(trans.obs, _acts, _next_obs)
                + self.config["gamma"] * self.module.value(_next_obs).squeeze(-1)
            )
        )

        loss = mle_loss + self.config["vf_loss_coeff"] * value_loss + svg_loss
        info = {
            "mle_loss": mle_loss.item(),
            "value_loss": value_loss.item(),
            "svg_loss": svg_loss.item(),
        }
        return loss, info

    @override(SVGInfTorchPolicy)
    def extra_grad_info(self):
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
        }
        return fetches
