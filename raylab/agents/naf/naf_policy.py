"""NAF policy class using PyTorch."""
import torch
import torch.nn as nn
from ray.rllib import SampleBatch
from ray.rllib.utils.annotations import override

from raylab.modules.catalog import get_module
import raylab.utils.pytorch as ptu
import raylab.policy as raypi


class NAFTorchPolicy(raypi.TargetNetworksMixin, raypi.TorchPolicy):
    """Normalized Advantage Function policy in Pytorch to use with RLlib."""

    # pylint: disable=abstract-method

    @staticmethod
    @override(raypi.TorchPolicy)
    def get_default_config():
        """Return the default config for NAF."""
        # pylint: disable=cyclic-import
        from raylab.agents.naf.naf import DEFAULT_CONFIG

        return DEFAULT_CONFIG

    @override(raypi.TorchPolicy)
    def make_module(self, obs_space, action_space, config):
        module_config = config["module"]
        module_config["type"] = "NAFModule"
        module_config["double_q"] = config["clipped_double_q"]
        module_config["perturbed_policy"] = (
            config["exploration_config"]["type"]
            == "raylab.utils.exploration.ParameterNoise"
        )

        return get_module(obs_space, action_space, module_config)

    @override(raypi.TorchPolicy)
    def make_optimizer(self):
        return ptu.build_optimizer(self.module.critics, self.config["torch_optimizer"])

    @override(raypi.TorchPolicy)
    def learn_on_batch(self, samples):
        batch_tensors = self._lazy_tensor_dict(samples)

        with self.optimizer.optimize():
            loss, info = self.compute_loss(batch_tensors, self.module, self.config)
            loss.backward()

        info.update(self.extra_grad_info())
        self.update_targets("vcritics", "target_vcritics")
        return self._learner_stats(info)

    def compute_loss(self, batch_tensors, module, config):
        """Compute the forward pass of NAF's loss function.

        Arguments:
            batch_tensors (UsageTrackingDict): Dictionary of experience batches that are
                lazily converted to tensors.
            module (nn.Module): The module of the policy
            config (dict): The policy's configuration

        Returns:
            A scalar tensor sumarizing the losses for this experience batch.
        """
        with torch.no_grad():
            target_values = self._compute_critic_targets(batch_tensors, module, config)

        obs = batch_tensors[SampleBatch.CUR_OBS]
        actions = batch_tensors[SampleBatch.ACTIONS]
        action_values = torch.cat([m(obs, actions) for m in module.critics], dim=-1)
        loss_fn = torch.nn.MSELoss()
        td_error = loss_fn(
            action_values, target_values.unsqueeze(-1).expand_as(action_values)
        )

        stats = {
            "q_mean": action_values.mean().item(),
            "q_max": action_values.max().item(),
            "q_min": action_values.min().item(),
            "td_error": td_error.item(),
        }
        return td_error, stats

    @staticmethod
    def _compute_critic_targets(batch_tensors, module, config):
        rewards = batch_tensors[SampleBatch.REWARDS]
        next_obs = batch_tensors[SampleBatch.NEXT_OBS]
        dones = batch_tensors[SampleBatch.DONES]

        next_vals, _ = torch.cat(
            [m(next_obs) for m in module.target_vcritics], dim=-1
        ).min(dim=-1)
        return torch.where(dones, rewards, rewards + config["gamma"] * next_vals)

    @torch.no_grad()
    def extra_grad_info(self):
        """Compute gradient norm for components."""
        return {
            "grad_norm": nn.utils.clip_grad_norm_(
                self.module.critics.parameters(), float("inf")
            ).item()
        }
