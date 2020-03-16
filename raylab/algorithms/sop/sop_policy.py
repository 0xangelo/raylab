"""SOP policy class using PyTorch."""
import collections

import torch
import torch.nn as nn
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override

import raylab.utils.pytorch as torch_util
import raylab.policy as raypi
from raylab.modules.catalog import get_module

OptimizerCollection = collections.namedtuple("OptimizerCollection", "policy critic")


class SOPTorchPolicy(
    raypi.AdaptiveParamNoiseMixin,
    raypi.PureExplorationMixin,
    raypi.TargetNetworksMixin,
    raypi.TorchPolicy,
):
    """Streamlined Off-Policy policy in PyTorch to use with RLlib."""

    # pylint: disable=abstract-method

    @staticmethod
    @override(raypi.TorchPolicy)
    def get_default_config():
        """Return the default configuration for SOP."""
        # pylint: disable=cyclic-import
        from raylab.algorithms.sop.sop import DEFAULT_CONFIG

        return DEFAULT_CONFIG

    @override(raypi.TorchPolicy)
    def make_module(self, obs_space, action_space, config):
        module_config = config["module"]
        for key in (
            "clipped_double_q",
            "exploration",
            "exploration_gaussian_sigma",
            "target_policy_smoothing",
            "target_gaussian_sigma",
        ):
            module_config[key] = config[key]

        return get_module(module_config["name"], obs_space, action_space, module_config)

    @override(raypi.TorchPolicy)
    def optimizer(self):
        pi_cls = torch_util.get_optimizer_class(self.config["policy_optimizer"]["name"])
        pi_optim = pi_cls(
            self.module.policy.parameters(),
            **self.config["policy_optimizer"]["options"]
        )

        qf_cls = torch_util.get_optimizer_class(self.config["critic_optimizer"]["name"])
        qf_optim = qf_cls(
            self.module.critics.parameters(),
            **self.config["critic_optimizer"]["options"]
        )

        return OptimizerCollection(policy=pi_optim, critic=qf_optim)

    @override(raypi.AdaptiveParamNoiseMixin)
    def _compute_noise_free_actions(self, sample_batch):
        obs_tensors = self.convert_to_tensor(sample_batch[SampleBatch.CUR_OBS])
        return self.module.policy[:-1](obs_tensors).numpy()

    @override(raypi.AdaptiveParamNoiseMixin)
    def _compute_noisy_actions(self, sample_batch):
        obs_tensors = self.convert_to_tensor(sample_batch[SampleBatch.CUR_OBS])
        return self.module.perturbed_policy[:-1](obs_tensors).numpy()

    @torch.no_grad()
    @override(raypi.TorchPolicy)
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

        if self.is_uniform_random:
            actions = self._uniform_random_actions(obs_batch)
        elif self.config["greedy"]:
            actions = self.module.policy(obs_batch)
        else:
            actions = self.module.sampler(obs_batch)

        return actions.cpu().numpy(), state_batches, {}

    @override(raypi.TorchPolicy)
    def learn_on_batch(self, samples):
        batch_tensors = self._lazy_tensor_dict(samples)

        info = {}
        info.update(self._update_critic(batch_tensors, self.module, self.config))
        info.update(self._update_policy(batch_tensors, self.module, self.config))

        self.update_targets("critics", "target_critics")
        return self._learner_stats(info)

    def _update_critic(self, batch_tensors, module, config):
        critic_loss, info = self.compute_critic_loss(batch_tensors, module, config)
        self._optimizer.critic.zero_grad()
        critic_loss.backward()
        grad_stats = {
            "critic_grad_norm": nn.utils.clip_grad_norm_(
                module.critics.parameters(), float("inf")
            )
        }
        info.update(grad_stats)

        self._optimizer.critic.step()
        return info

    def compute_critic_loss(self, batch_tensors, module, config):
        """Compute loss for Q value function."""
        obs = batch_tensors[SampleBatch.CUR_OBS]
        actions = batch_tensors[SampleBatch.ACTIONS]

        with torch.no_grad():
            target_values = self._compute_critic_targets(batch_tensors, module, config)
        loss_fn = nn.MSELoss()
        values = torch.cat([m(obs, actions) for m in module.critics], dim=-1)
        critic_loss = loss_fn(values, target_values.unsqueeze(-1).expand_as(values))

        stats = {
            "q_mean": values.mean().item(),
            "q_max": values.max().item(),
            "q_min": values.min().item(),
            "td_error": critic_loss.item(),
        }
        return critic_loss, stats

    @staticmethod
    def _compute_critic_targets(batch_tensors, module, config):
        rewards = batch_tensors[SampleBatch.REWARDS]
        next_obs = batch_tensors[SampleBatch.NEXT_OBS]
        dones = batch_tensors[SampleBatch.DONES]

        next_acts = module.target_policy(next_obs)
        next_vals, _ = torch.cat(
            [m(next_obs, next_acts) for m in module.target_critics], dim=-1
        ).min(dim=-1)
        return torch.where(dones, rewards, rewards + config["gamma"] * next_vals)

    def _update_policy(self, batch_tensors, module, config):
        policy_loss, info = self.compute_policy_loss(batch_tensors, module, config)
        self._optimizer.policy.zero_grad()
        policy_loss.backward()
        grad_stats = {
            "policy_grad_norm": nn.utils.clip_grad_norm_(
                module.policy.parameters(), float("inf")
            ),
            "param_noise_stddev": self.curr_param_stddev,
        }
        info.update(grad_stats)

        self._optimizer.policy.step()
        return info

    @staticmethod
    def compute_policy_loss(batch_tensors, module, config):
        """Compute loss for deterministic policy gradient."""
        # pylint: disable=unused-argument
        obs = batch_tensors[SampleBatch.CUR_OBS]

        actions = module.policy(obs)
        action_values, _ = torch.cat(
            [m(obs, actions) for m in module.critics], dim=-1
        ).min(dim=-1)
        max_objective = torch.mean(action_values)

        stats = {
            "policy_loss": max_objective.neg().item(),
            "qpi_mean": max_objective.item(),
        }
        return max_objective.neg(), stats
