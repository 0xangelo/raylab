"""SVG(inf) policy class using PyTorch."""
import collections

import torch
import torch.nn as nn
from ray.rllib.policy.policy import LEARNER_STATS_KEY
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override

from raylab.policy import TorchPolicy
from raylab.algorithms.svg.svg_inf_policy import SVGInfTorchPolicy
import raylab.modules as modules
import raylab.utils.pytorch as torch_util


Transition = collections.namedtuple("Transition", "obs actions rewards next_obs dones")


class SVGOneTorchPolicy(TorchPolicy):
    """Stochastic Value Gradients policy for full trajectories."""

    # pylint: disable=abstract-method

    ACTION_LOGP = "action_logp"

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)

        self.module = SVGInfTorchPolicy._make_module(  # pylint: disable=protected-access
            self.observation_space, self.action_space, self.config
        )
        self._optimizer = self.optimizer()

    @staticmethod
    @override(TorchPolicy)
    def get_default_config():
        """Return the default config for SVG(1)"""
        # pylint: disable=cyclic-import
        from raylab.algorithms.svg.svg_one import DEFAULT_CONFIG

        return DEFAULT_CONFIG

    @torch.no_grad()
    @override(TorchPolicy)
    def compute_actions(
        self,
        obs_batch,
        state_batches,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs
    ):
        # pylint: disable=too-many-arguments,unused-argument
        obs_batch = self.convert_to_tensor(obs_batch)
        dist_params = self.module.policy(obs_batch)
        actions = self.module.policy_rsample(dist_params)
        logp = self.module.policy_logp(dist_params, actions)

        extra_fetches = {self.ACTION_LOGP: logp.cpu().numpy()}
        return actions.cpu().numpy(), state_batches, extra_fetches

    @override(TorchPolicy)
    def learn_on_batch(self, samples):
        batch_tensors = self._lazy_tensor_dict(samples)
        loss, info = self.compute_joint_loss(batch_tensors)
        self._optimizer.zero_grad()
        loss.backward()
        info.update(self.extra_grad_info())
        self._optimizer.step()
        self.update_targets()

        return {LEARNER_STATS_KEY: info}

    @override(TorchPolicy)
    def optimizer(self):
        """Custom PyTorch optimizer to use."""
        optim_cls = torch_util.get_optimizer_class(self.config["torch_optimizer"])
        options = self.config["torch_optimizer_options"]
        params = [
            dict(params=self.module[k].parameters(), **options[k]) for k in options
        ]
        return optim_cls(params)

    # === NEW METHODS ===
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
            targets = self.module.target_value(trans.next_obs).squeeze(-1)
            dist_params = self.module.policy(trans.obs)
            curr_logp = self.module.policy_logp(dist_params, trans.actions)
            is_ratio = torch.exp(curr_logp - batch_tensors[self.ACTION_LOGP])
            is_ratio = torch.clamp(is_ratio, max=self.config["max_is_ratio"])

        targets = torch.where(
            trans.dones, trans.rewards, trans.rewards + self.config["gamma"] * targets
        )
        values = self.module.value(trans.obs).squeeze(-1)
        value_loss = torch.mean(
            is_ratio * nn.MSELoss(reduction="none")(values, targets) / 2
        )

        _acts = self.module.policy_rsample(self.module.policy(trans.obs), trans.actions)
        _next_obs = self.module.model_rsample(
            self.module.model(trans.obs, _acts), trans.next_obs
        )
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

    def update_targets(self):
        """Update target networks through one step of polyak averaging."""
        polyak = self.config["polyak"]
        torch_util.update_polyak(self.module.value, self.module.target_value, polyak)

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

    def add_kl_info(self, tensors):
        """Add average KL divergence between new and old policies."""
        keys = (SampleBatch.CUR_OBS, SampleBatch.ACTIONS, self.ACTION_LOGP)
        obs, act, logp = [tensors[k] for k in keys]
        dist_params = self.module.policy(obs)
        _logp = self.module.policy_logp(dist_params, act)

        return {"kl_div": torch.mean(logp - _logp).item()}
