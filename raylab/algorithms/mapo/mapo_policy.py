"""Policy for MAPO using PyTorch."""
import collections

import torch
import torch.nn as nn
from torch._six import inf
from ray.rllib.utils.annotations import override
from ray.rllib.policy.sample_batch import SampleBatch

import raylab.policy as raypi
import raylab.modules as mods
import raylab.utils.pytorch as torch_util
import raylab.algorithms.mapo.mapo_module as mapom


OptimizerCollection = collections.namedtuple(
    "OptimizerCollection", "policy critic model"
)


class MAPOTorchPolicy(
    raypi.AdaptiveParamNoiseMixin,
    raypi.PureExplorationMixin,
    raypi.TargetNetworksMixin,
    raypi.TorchPolicy,
):
    """Model-Aware Policy Optimization policy in PyTorch to use with RLlib."""

    # pylint: disable=abstract-method

    @staticmethod
    @override(raypi.TorchPolicy)
    def get_default_config():
        """Return the default configuration for MAPO."""
        # pylint: disable=cyclic-import
        from raylab.algorithms.mapo.mapo import DEFAULT_CONFIG

        return DEFAULT_CONFIG

    @override(raypi.TorchPolicy)
    def make_module(self, obs_space, action_space, config):
        module = nn.ModuleDict()
        module.update(self._make_policy(obs_space, action_space, config))

        def make_critic():
            return self._make_critic(obs_space, action_space, config)

        module.critics = nn.ModuleList([make_critic()])
        module.target_critics = nn.ModuleList([make_critic()])
        if config["clipped_double_q"]:
            module.critics.append(make_critic())
            module.target_critics.append(make_critic())
        module.target_critics.load_state_dict(module.critics.state_dict())

        module.update(self._make_model(obs_space, action_space, config))
        return module

    def _make_policy(self, obs_space, action_space, config):
        policy_config = config["module"]["policy"]

        def _make_modules():
            logits = mods.FullyConnected(
                in_features=obs_space.shape[0],
                units=policy_config["units"],
                activation=policy_config["activation"],
                layer_norm=policy_config.get(
                    "layer_norm", config["exploration"] == "parameter_noise"
                ),
                **policy_config["initializer_options"]
            )
            mu_ = mods.NormalizedLinear(
                in_features=logits.out_features,
                out_features=action_space.shape[0],
                beta=config["beta"],
            )
            squash = mods.TanhSquash(
                self.convert_to_tensor(action_space.low),
                self.convert_to_tensor(action_space.high),
            )
            return logits, mu_, squash

        logits_module, mu_module, squash_module = _make_modules()
        modules = {}
        modules["policy"] = nn.Sequential(logits_module, mu_module, squash_module)

        if config["exploration"] == "gaussian":
            expl_noise = mods.GaussianNoise(config["exploration_gaussian_sigma"])
            modules["sampler"] = nn.Sequential(
                logits_module, mu_module, expl_noise, squash_module
            )
        elif config["exploration"] == "parameter_noise":
            modules["sampler"] = modules["perturbed_policy"] = nn.Sequential(
                *_make_modules()
            )
        else:
            modules["sampler"] = modules["policy"]

        if config["target_policy_smoothing"]:
            modules["target_policy"] = nn.Sequential(
                logits_module,
                mu_module,
                mods.GaussianNoise(config["target_gaussian_sigma"]),
                squash_module,
            )
        else:
            modules["target_policy"] = modules["policy"]
        return modules

    @staticmethod
    def _make_critic(obs_space, action_space, config):
        critic_config = config["module"]["critic"]
        return mods.ActionValueFunction.from_scratch(
            obs_dim=obs_space.shape[0],
            action_dim=action_space.shape[0],
            delay_action=critic_config["delay_action"],
            units=critic_config["units"],
            activation=critic_config["activation"],
            **critic_config["initializer_options"]
        )

    @staticmethod
    def _make_model(obs_space, action_space, config):
        model_config = config["module"]["model"]
        model_module = mapom.DynamicsModel.from_scratch(
            obs_dim=obs_space.shape[0],
            action_dim=action_space.shape[0],
            input_dependent_scale=model_config["input_dependent_scale"],
            delay_action=model_config["delay_action"],
            units=model_config["units"],
            activation=model_config["activation"],
            **model_config["initializer_options"]
        )

        sampler_module = mapom.DynamicsModelRSample(model_module)
        logp_module = mapom.DynamicsModelLogProb(model_module)
        return {
            "model": model_module,
            "model_sampler": sampler_module,
            "model_logp": logp_module,
        }

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

        dm_cls = torch_util.get_optimizer_class(self.config["model_optimizer"]["name"])
        dm_optim = dm_cls(
            self.module.model.parameters(), **self.config["model_optimizer"]["options"]
        )

        return OptimizerCollection(policy=pi_optim, critic=qf_optim, model=dm_optim)

    def set_reward_fn(self, reward_fn):
        """Set the reward function to use when unrolling the policy and model."""
        self.module.reward = mods.Lambda(reward_fn)

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
        info.update(self._update_model(batch_tensors, self.module, self.config))
        info.update(self._update_policy(batch_tensors, self.module, self.config))

        self.update_targets("critics", "target_critics")
        return self._learner_stats(info)

    def _update_critic(self, batch_tensors, module, config):
        critic_loss, info = self.compute_critic_loss(batch_tensors, module, config)
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

    def _update_model(self, batch_tensors, module, config):
        model_loss, info = self.compute_model_loss(batch_tensors, module, config)
        self._optimizer.model.zero_grad()
        model_loss.backward()
        grad_stats = {
            "model_grad_norm": nn.utils.clip_grad_norm_(
                module.model.parameters(), float("inf")
            )
        }
        info.update(grad_stats)

        self._optimizer.model.step()
        return info

    def compute_model_loss(self, batch_tensors, module, config):
        """Compute policy gradient-aware (PGA) or MLE model loss."""
        with self.freeze_nets("model"):
            dpg_loss, dpg_info = self.compute_dpg_loss(batch_tensors, module, config)
            dpg_grads = torch.autograd.grad(dpg_loss, module.policy.parameters())

        madpg_loss, info = self.compute_madpg_loss(batch_tensors, module, config)
        madpg_grads = torch.autograd.grad(
            madpg_loss, module.policy.parameters(), create_graph=True
        )

        total_norm = self.compute_total_diff_norm(
            dpg_grads, madpg_grads, config["norm_type"]
        )

        info.update({"target_" + k: v for k, v in dpg_info.items()})
        return total_norm, info

    @staticmethod
    def compute_dpg_loss(batch_tensors, module, config):
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

    @staticmethod
    def compute_madpg_loss(batch_tensors, module, config):
        """Compute loss for model-aware deterministic policy gradient."""
        obs = batch_tensors[SampleBatch.CUR_OBS]
        actions = module.policy(obs)
        next_obs, logp = module.model_sampler(
            obs, actions, torch.as_tensor([config["num_model_samples"]])
        )
        rewards = module.reward(obs, actions, next_obs)
        with torch.no_grad():
            next_acts = module.policy(next_obs)
        next_vals, _ = torch.cat(
            [m(next_obs, next_acts) for m in module.critics], dim=-1
        ).min(dim=-1)

        loss = torch.mean(logp * (rewards + config["gamma"] * next_vals), dim=0).mean()
        return loss, {"model_aware_loss": loss.item()}

    @staticmethod
    def compute_total_diff_norm(atensors, btensors, norm_type):
        """Compute the norm of the difference of tensors as a flattened vector."""
        if norm_type == inf:
            total_norm = max((a - b).abs().max() for a, b in zip(atensors, btensors))
        else:
            total_norm = 0
            for atensor, btensor in zip(atensors, btensors):
                norm = (atensor - btensor).norm(norm_type)
                total_norm += norm ** norm_type
            total_norm = total_norm ** (1.0 / norm_type)
        return total_norm

    def _update_policy(self, batch_tensors, module, config):
        policy_loss, info = self.compute_madpg_loss(batch_tensors, module, config)
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
