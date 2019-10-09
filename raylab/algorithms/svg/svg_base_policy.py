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

    def set_reward_fn(self, reward_fn):
        """Set the reward function to use when unrolling the policy and model."""

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
        actions, logp = self.module.sampler(obs_batch)

        extra_fetches = {self.ACTION_LOGP: logp.cpu().numpy()}
        return actions.cpu().numpy(), state_batches, extra_fetches

    @override(TorchPolicy)
    def make_module(self, obs_space, action_space, config):
        module = nn.ModuleDict()
        module.update(self._make_model(obs_space, action_space, config))

        def make_vf():
            return self._make_critic(obs_space, config)

        module.value = make_vf()
        module.target_value = make_vf().requires_grad_(False)
        module.target_value.load_state_dict(module.value.state_dict())

        module.update(self._make_policy(obs_space, action_space, config))
        return module

    @staticmethod
    def _make_model(obs_space, action_space, config):
        model_config = config["module"]["model"]
        logits_modules = [
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
        model = svgm.ParallelDynamicsModel(*logits_modules)
        return {
            "model": model,
            "model_logp": svgm.ModelLogProb(
                model, mods.DiagMultivariateNormalLogProb()
            ),
            "model_reproduce": svgm.ModelReproduce(
                model, mods.DiagMultivariateNormalReproduce()
            ),
        }

    @staticmethod
    def _make_critic(obs_space, config):
        value_config = config["module"]["value"]
        logits_module = mods.FullyConnected(
            in_features=obs_space.shape[0],
            units=value_config["layers"],
            activation=value_config["activation"],
            **value_config["initializer_options"]
        )
        value_output = mods.ValueFunction(logits_module.out_features)
        return nn.Sequential(logits_module, value_output)

    def _make_policy(self, obs_space, action_space, config):
        policy_config = config["module"]["policy"]
        logits_module = mods.FullyConnected(
            in_features=obs_space.shape[0],
            units=policy_config["layers"],
            activation=policy_config["activation"],
            **policy_config["initializer_options"]
        )
        params_module = mods.DiagMultivariateNormalParams(
            logits_module.out_features,
            action_space.shape[0],
            input_dependent_scale=policy_config["input_dependent_scale"],
        )
        policy = nn.Sequential(logits_module, params_module)
        dist_params = dict(
            mean_only=config.get("mean_action_only", False),
            squashed=True,
            action_low=self.convert_to_tensor(action_space.low),
            action_high=self.convert_to_tensor(action_space.high),
        )
        return {
            "policy": policy,
            "sampler": nn.Sequential(
                policy, mods.DiagMultivariateNormalRSample(**dist_params)
            ),
            "policy_logp": svgm.PolicyLogProb(
                policy, mods.DiagMultivariateNormalLogProb(**dist_params)
            ),
            "policy_reproduce": svgm.PolicyReproduce(
                policy, mods.DiagMultivariateNormalReproduce(**dist_params)
            ),
        }

    @torch.no_grad()
    @override(AdaptiveKLCoeffMixin)
    def _kl_divergence(self, sample_batch):
        return self._avg_kl_divergence(self._lazy_tensor_dict(sample_batch)).item()

    def _avg_kl_divergence(self, batch_tensors):
        """Compute the empirical average KL divergence given sample tensors."""

    @torch.no_grad()
    def add_importance_sampling_ratios(self, batch_tensors):
        """Compute and add truncated importance sampling ratios to tensor batch."""
        is_ratios = self._compute_is_ratios(batch_tensors)
        _is_ratios = torch.clamp(is_ratios, max=self.config["max_is_ratio"])
        batch_tensors[self.IS_RATIOS] = _is_ratios
        return batch_tensors, {"avg_is_ratio": is_ratios.mean().item()}

    def _compute_is_ratios(self, batch_tensors):
        curr_logp = self.module.policy_logp(
            batch_tensors[SampleBatch.CUR_OBS], batch_tensors[SampleBatch.ACTIONS]
        )
        is_ratio = torch.exp(curr_logp - batch_tensors[self.ACTION_LOGP])
        return is_ratio

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

    def _avg_model_logp(self, batch_tensors):
        return self.module.model_logp(
            batch_tensors[SampleBatch.CUR_OBS],
            batch_tensors[SampleBatch.ACTIONS],
            batch_tensors[SampleBatch.NEXT_OBS],
        ).mean()

    def _compute_value_targets(self, batch_tensors):
        next_obs = batch_tensors[SampleBatch.NEXT_OBS]
        next_vals = self.module.target_value(next_obs).squeeze(-1)
        targets = torch.where(
            batch_tensors[SampleBatch.DONES],
            batch_tensors[SampleBatch.REWARDS],
            batch_tensors[SampleBatch.REWARDS] + self.config["gamma"] * next_vals,
        )
        return targets

    def update_targets(self):
        """Update target networks through one step of polyak averaging."""
        polyak = self.config["polyak"]
        torch_util.update_polyak(self.module.value, self.module.target_value, polyak)
