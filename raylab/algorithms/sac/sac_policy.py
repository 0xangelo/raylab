"""SAC policy class using PyTorch."""
import collections

import torch
import torch.nn as nn
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override

import raylab.modules as mods
import raylab.utils.pytorch as torch_util
from raylab.policy import TorchPolicy, PureExplorationMixin, TargetNetworksMixin


OptimizerCollection = collections.namedtuple(
    "OptimizerCollection", "policy critic alpha"
)


class SACTorchPolicy(PureExplorationMixin, TargetNetworksMixin, TorchPolicy):
    """Soft Actor-Critic policy in PyTorch to use with RLlib."""

    # pylint: disable=abstract-method

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        if self.config["target_entropy"] is None:
            self.config["target_entropy"] = -action_space.shape[0]

    @staticmethod
    @override(TorchPolicy)
    def get_default_config():
        """Return the default config for SAC."""
        # pylint: disable=cyclic-import
        from raylab.algorithms.sac.sac import DEFAULT_CONFIG

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

        module.log_alpha = nn.Parameter(torch.zeros([]))
        return module

    def _make_policy(self, obs_space, action_space, config):
        policy_config = config["module"]["policy"]
        logits_module = mods.FullyConnected(
            in_features=obs_space.shape[0],
            units=policy_config["units"],
            activation=policy_config["activation"],
            **policy_config["initializer_options"]
        )
        params_module = mods.DiagMultivariateNormalParams(
            logits_module.out_features,
            action_space.shape[0],
            input_dependent_scale=policy_config["input_dependent_scale"],
        )
        policy_module = nn.Sequential(logits_module, params_module)
        sampler_module = nn.Sequential(
            policy_module,
            mods.DiagMultivariateNormalRSample(
                mean_only=config["mean_action_only"],
                squashed=True,
                action_low=self.convert_to_tensor(action_space.low),
                action_high=self.convert_to_tensor(action_space.high),
            ),
        )
        return {"policy": policy_module, "sampler": sampler_module}

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

        al_cls = torch_util.get_optimizer_class(self.config["alpha_optimizer"]["name"])
        al_optim = al_cls(
            [self.module.log_alpha], **self.config["alpha_optimizer"]["options"]
        )

        return OptimizerCollection(policy=pi_optim, critic=qf_optim, alpha=al_optim)

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
            actions, _ = self.module.sampler(obs_batch)

        return actions.cpu().numpy(), state_batches, {}

    @override(TorchPolicy)
    def learn_on_batch(self, samples):
        batch_tensors = self._lazy_tensor_dict(samples)
        module, config = self.module, self.config
        info = {}

        info.update(self._update_critic(batch_tensors, module, config))
        info.update(self._update_policy(batch_tensors, module, config))
        info.update(self._update_alpha(batch_tensors, module, config))

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
        """Compute Soft Policy Iteration loss for Q value function."""
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

        next_acts, logp = module.sampler(next_obs)
        next_vals, _ = torch.cat(
            [m(next_obs, next_acts) for m in module.target_critics], dim=-1
        ).min(dim=-1)
        return torch.where(
            dones,
            rewards,
            rewards + config["gamma"] * (next_vals - module.log_alpha.exp() * logp),
        )

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
        apply_stats = {}
        info.update(apply_stats)
        return info

    @staticmethod
    def compute_policy_loss(batch_tensors, module, config):
        """Compute Soft Policy Iteration loss for reparameterized stochastic policy."""
        # pylint: disable=unused-argument
        obs = batch_tensors[SampleBatch.CUR_OBS]

        actions, logp = module.sampler(obs)
        action_values, _ = torch.cat(
            [m(obs, actions) for m in module.critics], dim=-1
        ).min(dim=-1)
        max_objective = torch.mean(action_values - module.log_alpha.exp() * logp)

        stats = {
            "policy_loss": max_objective.neg().item(),
            "qpi_mean": action_values.mean().item(),
            "logp_mean": logp.mean().item(),
        }
        return max_objective.neg(), stats

    def _update_alpha(self, batch_tensors, module, config):
        alpha_loss, info = self.compute_alpha_loss(batch_tensors, module, config)
        self._optimizer.alpha.zero_grad()
        alpha_loss.backward()
        grad_stats = {
            "alpha_grad_norm": self.module.log_alpha.grad.norm().item(),
            "curr_alpha": self.module.log_alpha.exp().item(),
        }
        info.update(grad_stats)

        self._optimizer.alpha.step()
        return info

    @staticmethod
    def compute_alpha_loss(batch_tensors, module, config):
        """Compute entropy coefficient loss."""
        with torch.no_grad():
            _, logp = module.sampler(batch_tensors[SampleBatch.CUR_OBS])
        alpha = module.log_alpha.exp()
        entropy_diff = torch.mean(-alpha * logp - alpha * config["target_entropy"])
        return entropy_diff, {"alpha_loss": entropy_diff.item()}
