"""SOP policy class using PyTorch."""
import collections

import torch
import torch.nn as nn
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override

import raylab.modules as mods
import raylab.utils.pytorch as torch_util
from raylab.policy import TorchPolicy, PureExplorationMixin, TargetNetworksMixin

OptimizerCollection = collections.namedtuple("OptimizerCollection", "policy critic")


class SOPTorchPolicy(PureExplorationMixin, TargetNetworksMixin, TorchPolicy):
    """Streamlined Off-Policy policy in PyTorch to use with RLlib."""

    # pylint: disable=abstract-method

    @staticmethod
    @override(TorchPolicy)
    def get_default_config():
        """Return the default configuration for SOP."""
        # pylint: disable=cyclic-import
        from raylab.algorithms.sop.sop import DEFAULT_CONFIG

        return DEFAULT_CONFIG

    @override(TorchPolicy)
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

        return module

    def _make_policy(self, obs_space, action_space, config):
        policy_config = config["module"]["policy"]
        logits_module = mods.FullyConnected(
            in_features=obs_space.shape[0],
            units=policy_config["units"],
            activation=policy_config["activation"],
            **policy_config["initializer_options"]
        )
        mu_module = mods.NormalizedLinear(
            in_features=logits_module.out_features,
            out_features=action_space.shape[0],
            beta=config["beta"],
        )
        squashing_module = mods.TanhSquash(
            self.convert_to_tensor(action_space.low),
            self.convert_to_tensor(action_space.high),
        )
        modules = {}
        modules["policy"] = nn.Sequential(logits_module, mu_module, squashing_module)
        modules["sampler"] = nn.Sequential(
            logits_module,
            mu_module,
            mods.GaussianNoise(
                config["exploration_gaussian_sigma"] if config["sampler_noise"] else 0
            ),
            squashing_module,
        )
        if config["target_policy_smoothing"]:
            modules["target_action"] = nn.Sequential(
                logits_module,
                mu_module,
                mods.GaussianNoise(config["target_gaussian_sigma"]),
                squashing_module,
            )
        else:
            modules["target_action"] = modules["policy"]
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

    @override(TorchPolicy)
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

        if self.is_uniform_random:
            actions = self._uniform_random_actions(obs_batch)
        else:
            actions = self.module.sampler(obs_batch)

        return actions.cpu().numpy(), state_batches, {}

    @override(TorchPolicy)
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

        next_acts = module.target_action(next_obs)
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
            )
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
            "qpi_mean": action_values.mean().item(),
        }
        return max_objective.neg(), stats
