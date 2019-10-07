"""SAC policy class using PyTorch."""
import itertools
import collections

import torch
import torch.nn as nn
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override

import raylab.modules as mods
import raylab.utils.pytorch as torch_util
from raylab.policy import TorchPolicy, PureExplorationMixin
from raylab.algorithms.sac.sac_module import ActionValueFunction


OptimizerCollection = collections.namedtuple(
    "OptimizerCollection", "policy critic alpha"
)


class SACTorchPolicy(PureExplorationMixin, TorchPolicy):
    """Soft Actor-Critic policy in PyTorch to use with RLlib."""

    # pylint: disable=abstract-method

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self._optimizer = self.optimizer()
        if self.config["target_entropy"] is None:
            self.config["target_entropy"] = -len(action_space.shape)

    @staticmethod
    @override(TorchPolicy)
    def get_default_config():
        """Return the default config for SAC."""
        # pylint: disable=cyclic-import
        from raylab.algorithms.sac.sac import DEFAULT_CONFIG

        return DEFAULT_CONFIG

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

        critic_loss, stats = self.compute_critic_loss(batch_tensors, module, config)
        info.update(stats)
        self._optimizer.critic.zero_grad()
        critic_loss.backward()
        info.update(self.extra_grad_info(batch_tensors, "critic"))
        self._optimizer.critic.step()
        info.update(self.extra_apply_info(batch_tensors, "critic"))

        policy_loss, stats = self.compute_policy_loss(batch_tensors, module, config)
        info.update(stats)
        self._optimizer.policy.zero_grad()
        policy_loss.backward()
        info.update(self.extra_grad_info(batch_tensors, "policy"))
        self._optimizer.policy.step()
        info.update(self.extra_apply_info(batch_tensors, "policy"))

        alpha_loss, stats = self.compute_alpha_loss(batch_tensors, module, config)
        info.update(stats)
        self._optimizer.alpha.zero_grad()
        alpha_loss.backward()
        info.update(self.extra_grad_info(batch_tensors, "alpha"))
        self._optimizer.alpha.step()
        info.update(self.extra_apply_info(batch_tensors, "alpha"))

        self.update_targets()
        losses_info = {
            "critic_loss": critic_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha_loss": alpha_loss.item(),
        }
        info.update(losses_info)
        return self._learner_stats(info)

    @override(TorchPolicy)
    def optimizer(self):
        pi_cls = torch_util.get_optimizer_class(self.config["policy_optimizer"]["name"])
        pi_optim = pi_cls(
            self.module.policy.parameters(),
            **self.config["policy_optimizer"]["options"]
        )

        qf_cls = torch_util.get_optimizer_class(self.config["critic_optimizer"]["name"])
        qf_params = self.module.critic.parameters()
        if self.config["clipped_double_q"]:
            qf_params = itertools.chain(qf_params, self.module.twin_critic.parameters())
        qf_optim = qf_cls(qf_params, **self.config["critic_optimizer"]["options"])

        al_cls = torch_util.get_optimizer_class(self.config["alpha_optimizer"]["name"])
        al_optim = al_cls(
            [self.module.log_alpha], **self.config["alpha_optimizer"]["options"]
        )

        return OptimizerCollection(policy=pi_optim, critic=qf_optim, alpha=al_optim)

    @override(TorchPolicy)
    def make_module(self, obs_space, action_space, config):
        module = nn.ModuleDict()
        module.update(self._make_policy(obs_space, action_space, config))

        def make_critic():
            return self._make_critic(obs_space, action_space, config)

        module.critic = make_critic()
        module.target_critic = make_critic()
        module.target_critic.load_state_dict(module.critic.state_dict())
        if config["clipped_double_q"]:
            module.twin_critic = make_critic()
            module.target_twin_critic = make_critic()
            module.target_twin_critic.load_state_dict(module.twin_critic.state_dict())

        module.log_alpha = nn.Parameter(torch.zeros([]))
        return module

    # ================================= NEW METHODS ====================================

    def compute_critic_loss(self, batch_tensors, module, config):
        """Compute Soft Policy Iteration loss for Q value function."""
        obs = batch_tensors[SampleBatch.CUR_OBS]
        actions = batch_tensors[SampleBatch.ACTIONS]
        with torch.no_grad():
            target_values = self._compute_critic_targets(batch_tensors, module, config)

        loss_fn = nn.MSELoss()
        values = module.critic(obs, actions).squeeze(-1)
        critic_loss = loss_fn(values, target_values)
        if config["clipped_double_q"]:
            twin_values = module.twin_critic(obs, actions).squeeze(-1)
            critic_loss = (critic_loss + loss_fn(twin_values, target_values)) / 2
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
        next_vals = module.target_critic(next_obs, next_acts).squeeze(-1)
        if config["clipped_double_q"]:
            twin_next_vals = module.target_twin_critic(next_obs, next_acts).squeeze(-1)
            next_vals = torch.min(next_vals, twin_next_vals)

        return torch.where(
            dones,
            rewards,
            rewards + config["gamma"] * (next_vals - module.log_alpha.exp() * logp),
        )

    @staticmethod
    def compute_policy_loss(batch_tensors, module, config):
        """Compute Soft Policy Iteration loss for reparameterized stochastic policy."""
        obs = batch_tensors[SampleBatch.CUR_OBS]
        actions, logp = module.sampler(obs)
        action_values = module.critic(obs, actions)
        if config["clipped_double_q"]:
            action_values = torch.min(action_values, module.twin_critic(obs, actions))
        max_objective = torch.mean(action_values - module.log_alpha.exp() * logp)
        stats = {
            "qpi_mean": action_values.mean().item(),
            "logp_mean": logp.mean().item(),
        }
        return max_objective.neg(), stats

    @staticmethod
    def compute_alpha_loss(batch_tensors, module, config):
        """Compute entropy coefficient loss."""
        _, logp = module.sampler(batch_tensors[SampleBatch.CUR_OBS])
        alpha = module.log_alpha.exp()
        entropy_diff = torch.mean(-alpha * logp - alpha * config["target_entropy"])
        return entropy_diff, {}

    @torch.no_grad()
    def extra_grad_info(self, batch_tensors, key):  # pylint: disable=unused-argument
        """Return extra stats for `key` component after computing its gradients."""
        if key == "critic":
            info = {
                "critic_grad_norm": nn.utils.clip_grad_norm_(
                    self._optimizer.critic.param_groups[0]["params"], float("inf")
                )
            }
        elif key == "policy":
            info = {
                "policy_grad_norm": nn.utils.clip_grad_norm_(
                    self.module.policy.parameters(), float("inf")
                )
            }
        elif key == "alpha":
            info = {
                "alpha_grad_norm": self.module.log_alpha.norm().item(),
                "curr_alpha": self.module.log_alpha.item(),
            }
        else:
            info = {}
        return info

    @torch.no_grad()
    def extra_apply_info(self, batch_tensors, key):
        """Return extra stats for `key` component after applying its gradients."""
        # pylint: disable=unused-argument,no-self-use
        if key == "policy":
            info = {}  # Calculate cross-entropy later
        else:
            info = {}
        return info

    def update_targets(self):
        """Update target networks through one step of polyak averaging."""
        polyak = self.config["polyak"]
        torch_util.update_polyak(self.module.critic, self.module.target_critic, polyak)
        if self.config["clipped_double_q"]:
            torch_util.update_polyak(
                self.module.twin_critic, self.module.target_twin_critic, polyak
            )

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
        logits_module = mods.StateActionEncoder(
            obs_dim=obs_space.shape[0],
            action_dim=action_space.shape[0],
            delay_action=critic_config["delay_action"],
            units=critic_config["units"],
            activation=critic_config["activation"],
            **critic_config["initializer_options"]
        )
        value_module = mods.ValueFunction(logits_module.out_features)
        return ActionValueFunction(logits_module, value_module)
