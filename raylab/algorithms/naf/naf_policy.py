"""NAF policy class using PyTorch."""
import torch
import torch.nn as nn
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override

import raylab.utils.pytorch as torch_util
import raylab.algorithms.naf.naf_module as mods
from raylab.policy import TorchPolicy, AdaptiveParamNoiseMixin, PureExplorationMixin


class NAFTorchPolicy(AdaptiveParamNoiseMixin, PureExplorationMixin, TorchPolicy):
    """Normalized Advantage Function policy in Pytorch to use with RLlib."""

    # pylint: disable=abstract-method

    @staticmethod
    @override(TorchPolicy)
    def get_default_config():
        """Return the default config for NAF."""
        # pylint: disable=cyclic-import
        from raylab.algorithms.naf.naf import DEFAULT_CONFIG

        return DEFAULT_CONFIG

    @override(TorchPolicy)
    def make_module(self, obs_space, action_space, config):
        module = nn.ModuleDict()
        module.update(self._make_naf(obs_space, action_space, config))
        if config["clipped_double_q"]:
            twin_modules = self._make_naf(obs_space, action_space, config)
            module["naf"].extend(twin_modules["naf"])
            module["value"].extend(twin_modules["value"])
        module.target_value = nn.ModuleList(
            [self._make_value(obs_space, config) for _ in module.value]
        )
        module.target_value.load_state_dict(module.value.state_dict())

        # Configure policy module based on exploration strategy
        if config["exploration"] == "full_gaussian":
            module.policy = mods.MultivariateGaussianPolicy(
                module.logits, module.action, module.tril
            )
        elif config["exploration"] == "parameter_noise":
            logits_module = self._make_encoder(obs_space, config)
            module.policy = nn.Sequential(
                logits_module,
                mods.ActionOutput(
                    in_features=logits_module.out_features,
                    action_low=self.convert_to_tensor(action_space.low),
                    action_high=self.convert_to_tensor(action_space.high),
                ),
            )
            module.target_policy = nn.Sequential(module.logits, module.action)
        else:
            module.policy = nn.Sequential(module.logits, module.action)

        return module

    def _make_naf(self, obs_space, action_space, config):
        value = self._make_value(obs_space, config)
        logits_module = value[0]
        value_module = value[1]
        action_module = mods.ActionOutput(
            in_features=logits_module.out_features,
            action_low=self.convert_to_tensor(action_space.low),
            action_high=self.convert_to_tensor(action_space.high),
        )
        tril_module = mods.TrilMatrix(logits_module.out_features, action_space.shape[0])
        advantage_module = mods.AdvantageFunction(tril_module, action_module)
        return {
            "naf": nn.ModuleList(
                [mods.NAF(logits_module, value_module, advantage_module)]
            ),
            "value": nn.ModuleList([value]),
            "logits": logits_module,
            "action": action_module,
            "tril": tril_module,
        }

    def _make_value(self, obs_space, config):
        logits_module = self._make_encoder(obs_space, config)
        value_module = mods.ValueFunction(logits_module.out_features)
        return nn.Sequential(logits_module, value_module)

    @staticmethod
    def _make_encoder(obs_space, config):
        return mods.FullyConnected(
            in_features=obs_space.shape[0],
            units=config["module"]["units"],
            activation=config["module"]["activation"],
            layer_norm=(config["exploration"] == "parameter_noise"),
            **config["module"]["initializer_options"],
        )

    @override(TorchPolicy)
    def optimizer(self):
        cls = torch_util.get_optimizer_class(self.config["torch_optimizer"])
        options = self.config["torch_optimizer_options"]
        return cls(self.module.naf.parameters(), **options)

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
        **kwargs,
    ):
        # pylint: disable=too-many-arguments,unused-argument
        obs_batch = self.convert_to_tensor(obs_batch)

        if self.config["greedy"]:
            actions = self._greedy_actions(obs_batch)
        elif self.is_uniform_random:
            actions = self._uniform_random_actions(obs_batch)
        elif self.config["exploration"] == "full_gaussian":
            actions = self._multivariate_gaussian_actions(obs_batch)
        elif self.config["exploration"] == "diag_gaussian":
            actions = self._diagonal_gaussian_actions(obs_batch)
        else:
            actions = self.module["policy"](obs_batch)

        return actions.cpu().numpy(), state_batches, {}

    # === Action Sampling ===
    def _greedy_actions(self, obs_batch):
        policy = self.module.policy
        if "target_policy" in self.module:
            policy = self.module.target_policy

        out = policy(obs_batch)
        if isinstance(out, tuple):
            out, _ = out
        return out

    def _multivariate_gaussian_actions(self, obs_batch):
        loc, scale_tril = self.module.policy(obs_batch)
        scale_coeff = self.config["scale_tril_coeff"]
        dist = torch.distributions.MultivariateNormal(
            loc=loc, scale_tril=scale_tril * scale_coeff
        )
        actions = dist.sample()
        return actions

    def _diagonal_gaussian_actions(self, obs_batch):
        loc = self.module.policy(obs_batch)
        stddev = self.config["diag_gaussian_stddev"]
        dist = torch.distributions.MultivariateNormal(
            loc=loc, scale_tril=torch.diag(torch.ones(self.action_space.shape)) * stddev
        )
        actions = dist.sample()
        return actions

    @override(AdaptiveParamNoiseMixin)
    def _compute_noise_free_actions(self, obs_batch):
        return self.module.target_policy(self.convert_to_tensor(obs_batch)).numpy()

    @torch.no_grad()
    @override(TorchPolicy)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        sample_batch = super().postprocess_trajectory(
            sample_batch, other_agent_batches=other_agent_batches, episode=episode
        )
        if self.config["exploration"] == "parameter_noise":
            self.update_parameter_noise(sample_batch)
        return sample_batch

    @override(TorchPolicy)
    def learn_on_batch(self, samples):
        batch_tensors = self._lazy_tensor_dict(samples)

        loss, info = self.compute_loss(batch_tensors, self.module, self.config)
        self._optimizer.zero_grad()
        loss.backward()
        info.update(self.extra_grad_info())
        self._optimizer.step()

        torch_util.update_polyak(
            self.module.value, self.module.target_value, self.config["polyak"]
        )
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
            )
        }
