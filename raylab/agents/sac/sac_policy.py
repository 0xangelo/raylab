"""SAC policy class using PyTorch."""
import collections

import torch
import torch.nn as nn
from ray.rllib import SampleBatch
from ray.rllib.utils.annotations import override

import raylab.utils.pytorch as ptu
from raylab.policy import TorchPolicy, TargetNetworksMixin


class SACTorchPolicy(TargetNetworksMixin, TorchPolicy):
    """Soft Actor-Critic policy in PyTorch to use with RLlib."""

    # pylint: disable=abstract-method

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        if self.config["target_entropy"] == "auto":
            self.config["target_entropy"] = -action_space.shape[0]

    @staticmethod
    @override(TorchPolicy)
    def get_default_config():
        """Return the default config for SAC."""
        # pylint: disable=cyclic-import
        from raylab.agents.sac.sac import DEFAULT_CONFIG

        return DEFAULT_CONFIG

    @override(TorchPolicy)
    def make_module(self, obs_space, action_space, config):
        module_config = config["module"]
        module_config.setdefault("critic", {})
        module_config["critic"]["double_q"] = config["clipped_double_q"]
        return super().make_module(obs_space, action_space, config)

    @override(TorchPolicy)
    def make_optimizer(self):
        config = self.config["torch_optimizer"]
        components = "actor critics alpha".split()

        optims = {k: ptu.build_optimizer(self.module[k], config[k]) for k in components}
        return collections.namedtuple("OptimizerCollection", components)(**optims)

    @override(TorchPolicy)
    def learn_on_batch(self, samples):
        batch_tensors = self._lazy_tensor_dict(samples)
        module, config = self.module, self.config
        info = {}

        info.update(self._update_critic(batch_tensors, module, config))
        info.update(self._update_actor(batch_tensors, module, config))
        if config["target_entropy"] is not None:
            info.update(self._update_alpha(batch_tensors, module, config))

        self.update_targets("critics", "target_critics")
        return self._learner_stats(info)

    def _update_critic(self, batch_tensors, module, config):
        with self.optimizer.critics.optimize():
            critic_loss, info = self.compute_critic_loss(batch_tensors, module, config)
            critic_loss.backward()

        info.update(self.extra_grad_info("critics", batch_tensors))
        return info

    def compute_critic_loss(self, batch_tensors, module, config):
        """Compute Soft Policy Iteration loss for Q value function."""
        obs = batch_tensors[SampleBatch.CUR_OBS]
        actions = batch_tensors[SampleBatch.ACTIONS]

        with torch.no_grad():
            target_values = self._compute_critic_targets(batch_tensors, module, config)
        loss_fn = nn.MSELoss()
        values = torch.cat([m(obs, actions) for m in module.critics], dim=-1)
        critic_loss = loss_fn(values, target_values.unsqueeze(-1).expand_as(values))

        info = {
            "q_mean": values.mean().item(),
            "q_max": values.max().item(),
            "q_min": values.min().item(),
            "loss(critics)": critic_loss.item(),
        }
        return critic_loss, info

    @staticmethod
    def _compute_critic_targets(batch_tensors, module, config):
        rewards = batch_tensors[SampleBatch.REWARDS]
        next_obs = batch_tensors[SampleBatch.NEXT_OBS]
        dones = batch_tensors[SampleBatch.DONES]

        next_acts, logp = module.actor.rsample(next_obs)
        next_vals, _ = torch.cat(
            [m(next_obs, next_acts) for m in module.target_critics], dim=-1
        ).min(dim=-1)
        return torch.where(
            dones,
            rewards,
            rewards + config["gamma"] * (next_vals - module.alpha() * logp),
        )

    def _update_actor(self, batch_tensors, module, config):
        with self.optimizer.actor.optimize():
            actor_loss, info = self.compute_actor_loss(batch_tensors, module, config)
            actor_loss.backward()

        info.update(self.extra_grad_info("actor", batch_tensors))
        return info

    @staticmethod
    def compute_actor_loss(batch_tensors, module, config):
        """Compute Soft Policy Iteration loss for reparameterized stochastic policy."""
        # pylint: disable=unused-argument
        obs = batch_tensors[SampleBatch.CUR_OBS]

        actions, logp = module.actor.rsample(obs)
        action_values, _ = torch.cat(
            [m(obs, actions) for m in module.critics], dim=-1
        ).min(dim=-1)
        max_objective = torch.mean(action_values - module.alpha() * logp)

        info = {
            "loss(actor)": max_objective.neg().item(),
            "qpi_mean": action_values.mean().item(),
            "entropy": -logp.mean().item(),
        }
        return max_objective.neg(), info

    def _update_alpha(self, batch_tensors, module, config):
        with self.optimizer.alpha.optimize():
            alpha_loss, info = self.compute_alpha_loss(batch_tensors, module, config)
            alpha_loss.backward()

        info.update(self.extra_grad_info("alpha", batch_tensors))
        return info

    @staticmethod
    def compute_alpha_loss(batch_tensors, module, config):
        """Compute entropy coefficient loss."""
        with torch.no_grad():
            _, logp = module.actor.rsample(batch_tensors[SampleBatch.CUR_OBS])
        alpha = module.alpha()
        entropy_diff = torch.mean(-alpha * logp - alpha * config["target_entropy"])
        info = {"loss(alpha)": entropy_diff.item(), "curr_alpha": alpha.item()}
        return entropy_diff, info

    @torch.no_grad()
    def extra_grad_info(self, component, batch_tensors):
        """Return statistics right after components are updated."""
        # pylint:disable=unused-argument
        return {
            f"grad_norm({component})": nn.utils.clip_grad_norm_(
                self.module[component].parameters(), float("inf")
            ).item()
        }
