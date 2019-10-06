"""Base Policy with common methods for all SVG variations."""
import torch
import torch.nn as nn
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override

from raylab.policy import TorchPolicy, AdaptiveKLCoeffMixin
import raylab.algorithms.svg.svg_module as svgm
import raylab.modules as mods
import raylab.utils.pytorch as torch_util


class SVGBaseTorchPolicy(AdaptiveKLCoeffMixin, TorchPolicy):
    """Stochastic Value Gradients policy using PyTorch."""

    # pylint: disable=abstract-method

    ACTION_LOGP = "action_logp"
    IS_RATIOS = "is_ratios"

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self.module = self._make_module(
            self.observation_space, self.action_space, self.config
        )

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

        extra_fetches = {
            self.ACTION_LOGP: logp.cpu().numpy(),
            **{k: v.numpy() for k, v in dist_params.items()},
        }
        return actions.cpu().numpy(), state_batches, extra_fetches

    @torch.no_grad()
    @override(AdaptiveKLCoeffMixin)
    def _kl_divergence(self, sample_batch):
        return self._avg_kl_divergence(self._lazy_tensor_dict(sample_batch)).item()

    # ================================= NEW METHODS ====================================

    def set_reward_fn(self, reward_fn):
        """Set the reward function to use when unrolling the policy and model."""

    @torch.no_grad()
    def add_importance_sampling_ratios(self, batch_tensors):
        """Compute and add truncated importance sampling ratios to tensor batch."""
        is_ratios = self._compute_is_ratios(batch_tensors)
        _is_ratios = torch.clamp(is_ratios, max=self.config["max_is_ratio"])
        batch_tensors[self.IS_RATIOS] = _is_ratios
        return batch_tensors, {"avg_is_ratio": is_ratios.mean().item()}

    def compute_joint_model_value_loss(self, batch_tensors):
        """Compute model MLE loss and fitted value function loss."""
        mle_loss = self._avg_model_logp(batch_tensors).neg()

        with torch.no_grad():
            targets = self._compute_value_targets(batch_tensors)
        _is_ratios = batch_tensors[self.IS_RATIOS]
        values = self.module.value(batch_tensors[SampleBatch.CUR_OBS]).squeeze(-1)
        value_loss = torch.mean(
            _is_ratios * nn.MSELoss(reduction="none")(values, targets) / 2
        )

        loss = mle_loss + self.config["vf_loss_coeff"] * value_loss
        return loss, {"mle_loss": mle_loss.item(), "value_loss": value_loss.item()}

    @staticmethod
    def _make_module(obs_space, action_space, config):
        module = nn.ModuleDict()
        module.update(SVGBaseTorchPolicy._make_model(obs_space, action_space, config))
        module.value = SVGBaseTorchPolicy._make_critic(obs_space, config)
        module.target_value = SVGBaseTorchPolicy._make_critic(obs_space, config)
        module.target_value.load_state_dict(module.value.state_dict())
        module.update(SVGBaseTorchPolicy._make_policy(obs_space, action_space, config))
        return module

    @staticmethod
    def _make_model(obs_space, action_space, config):
        model_config = config["module"]["model"]
        model_logits_modules = [
            mods.StateActionEncoder(
                obs_dim=obs_space.shape[0],
                action_dim=action_space.shape[0],
                delay_action=model_config["delay_action"],
                units=model_config["layers"],
                activation=model_config["activation"],
                **model_config["initializer_options"]
            )
            for _ in range(obs_space.shape[0])
        ]
        return {
            "model": svgm.ParallelDynamicsModel(*model_logits_modules),
            "model_logp": svgm.DiagNormalLogProb(),
            "model_rsample": svgm.DiagNormalRSample(),
        }

    @staticmethod
    def _make_critic(obs_space, config):
        value_config = config["module"]["value"]
        value_logits_module = mods.FullyConnected(
            in_features=obs_space.shape[0],
            units=value_config["layers"],
            activation=value_config["activation"],
            **value_config["initializer_options"]
        )
        value_output = mods.ValueFunction(value_logits_module.out_features)
        return nn.Sequential(value_logits_module, value_output)

    @staticmethod
    def _make_policy(obs_space, action_space, config):
        policy_config = config["module"]["policy"]
        policy_logits_module = mods.FullyConnected(
            in_features=obs_space.shape[0],
            units=policy_config["layers"],
            activation=policy_config["activation"],
            **policy_config["initializer_options"]
        )
        policy_dist_param_module = mods.DiagMultivariateNormalParams(
            policy_logits_module.out_features,
            action_space.shape[0],
            input_dependent_scale=policy_config["input_dependent_scale"],
        )

        return {
            "policy": nn.Sequential(policy_logits_module, policy_dist_param_module),
            "policy_logp": svgm.DiagNormalLogProb(),
            "policy_rsample": svgm.DiagNormalRSample(),
            "entropy": svgm.DiagNormalEntropy(),
            "kl_div": svgm.DiagNormalKL(),
        }

    def update_targets(self):
        """Update target networks through one step of polyak averaging."""
        polyak = self.config["polyak"]
        torch_util.update_polyak(self.module.value, self.module.target_value, polyak)

    def _avg_model_logp(self, batch_tensors):
        dist_params = self.module.model(
            batch_tensors[SampleBatch.CUR_OBS], batch_tensors[SampleBatch.ACTIONS]
        )
        residual = (
            batch_tensors[SampleBatch.NEXT_OBS] - batch_tensors[SampleBatch.CUR_OBS]
        )
        return self.module.model_logp(dist_params, residual).mean()

    def _compute_value_targets(self, batch_tensors):
        next_obs = batch_tensors[SampleBatch.NEXT_OBS]
        next_vals = self.module.target_value(next_obs).squeeze(-1)
        targets = torch.where(
            batch_tensors[SampleBatch.DONES],
            batch_tensors[SampleBatch.REWARDS],
            batch_tensors[SampleBatch.REWARDS] + self.config["gamma"] * next_vals,
        )
        return targets

    def _compute_is_ratios(self, batch_tensors):
        dist_params = self.module.policy(batch_tensors[SampleBatch.CUR_OBS])
        curr_logp = self.module.policy_logp(
            dist_params, batch_tensors[SampleBatch.ACTIONS]
        )
        is_ratio = torch.exp(curr_logp - batch_tensors[self.ACTION_LOGP])
        return is_ratio

    def _avg_kl_divergence(self, batch_tensors):
        new_params = self.module.policy(batch_tensors[SampleBatch.CUR_OBS])
        return self.module.kl_div(batch_tensors, new_params).mean()
