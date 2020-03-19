"""NAF policy class using PyTorch."""
import torch
import torch.nn as nn
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override

from raylab.modules.catalog import get_module
import raylab.utils.pytorch as torch_util
import raylab.policy as raypi


class NAFTorchPolicy(
    raypi.AdaptiveParamNoiseMixin,
    raypi.PureExplorationMixin,
    raypi.TargetNetworksMixin,
    raypi.TorchPolicy,
):
    """Normalized Advantage Function policy in Pytorch to use with RLlib."""

    # pylint: disable=abstract-method

    @staticmethod
    @override(raypi.TorchPolicy)
    def get_default_config():
        """Return the default config for NAF."""
        # pylint: disable=cyclic-import
        from raylab.algorithms.naf.naf import DEFAULT_CONFIG

        return DEFAULT_CONFIG

    @override(raypi.TorchPolicy)
    def make_module(self, obs_space, action_space, config):
        module_config = config["module"]
        for key in ("exploration", "clipped_double_q", "diag_gaussian_stddev"):
            module_config[key] = config[key]

        module_name = module_config["name"]
        assert (
            module_name == "NormalizedAdvantageFunction"
        ), "Incompatible module type f{module_name}"
        module = get_module(module_name, obs_space, action_space, module_config)
        return torch.jit.script(module) if module_config["torch_script"] else module

    @override(raypi.TorchPolicy)
    def optimizer(self):
        cls = torch_util.get_optimizer_class(self.config["torch_optimizer"]["name"])
        options = self.config["torch_optimizer"]["options"]
        return cls(self.module.naf.parameters(), **options)

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
        **kwargs,
    ):
        # pylint: disable=too-many-arguments,unused-argument
        obs_batch = self.convert_to_tensor(obs_batch)

        if self.config["greedy"]:
            actions = self.module.policy(obs_batch)
        elif self.is_uniform_random:
            actions = self._uniform_random_actions(obs_batch)
        else:
            actions = self.module.sampler(obs_batch)

        return actions.cpu().numpy(), state_batches, {}

    @override(raypi.AdaptiveParamNoiseMixin)
    def _compute_noise_free_actions(self, sample_batch):
        obs_tensors = self.convert_to_tensor(sample_batch[SampleBatch.CUR_OBS])
        return self.module.policy[:-1](obs_tensors).numpy()

    @override(raypi.AdaptiveParamNoiseMixin)
    def _compute_noisy_actions(self, sample_batch):
        obs_tensors = self.convert_to_tensor(sample_batch[SampleBatch.CUR_OBS])
        return self.module.perturbed_policy[:-1](obs_tensors).numpy()

    @override(raypi.TorchPolicy)
    def learn_on_batch(self, samples):
        batch_tensors = self._lazy_tensor_dict(samples)

        loss, info = self.compute_loss(batch_tensors, self.module, self.config)
        self._optimizer.zero_grad()
        loss.backward()
        info.update(self.extra_grad_info())
        self._optimizer.step()

        self.update_targets("value", "target_value")
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
        action_values = torch.cat([m(obs, actions) for m in module.naf], dim=-1)
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
            [m(next_obs) for m in module.target_value], dim=-1
        ).min(dim=-1)
        return torch.where(dones, rewards, rewards + config["gamma"] * next_vals)

    @torch.no_grad()
    def extra_grad_info(self):
        """Compute gradient norm for components."""
        return {
            "grad_norm": nn.utils.clip_grad_norm_(
                self.module.naf.parameters(), float("inf")
            ),
            "param_noise_stddev": self.curr_param_stddev,
        }
