"""Policy for MAPO using PyTorch."""
import collections

import torch
import torch.nn as nn
from ray.rllib.utils.annotations import override
from ray.rllib import SampleBatch

import raylab.policy as raypi
from raylab.envs.rewards import get_reward_fn
import raylab.utils.pytorch as ptu


class MAPOTorchPolicy(raypi.TargetNetworksMixin, raypi.TorchPolicy):
    """Model-Aware Policy Optimization policy in PyTorch to use with RLlib."""

    # pylint: disable=abstract-method

    def __init__(self, observation_space, action_space, config):
        assert (
            config.get("module", {}).get("torch_script", False) is False
        ), "MAPO uses operations incompatible with TorchScript."
        super().__init__(observation_space, action_space, config)
        self.reward = get_reward_fn(self.config["env"], self.config["env_config"])
        self.transition = None

    @staticmethod
    @override(raypi.TorchPolicy)
    def get_default_config():
        """Return the default configuration for MAPO."""
        # pylint: disable=cyclic-import
        from raylab.agents.mapo.mapo import DEFAULT_CONFIG

        return DEFAULT_CONFIG

    @override(raypi.TorchPolicy)
    def make_module(self, obs_space, action_space, config):
        module_config = config["module"]
        module_config.setdefault("critic", {})
        module_config["critic"]["double_q"] = config["clipped_double_q"]
        module_config.setdefault("actor", {})
        module_config["actor"]["perturbed_policy"] = (
            config["exploration_config"]["type"]
            == "raylab.utils.exploration.ParameterNoise"
        )
        # pylint:disable=no-member
        module = super().make_module(obs_space, action_space, config)
        # pylint:enable=no-member
        self.check_model(module.model.rsample)
        return module

    @override(raypi.TorchPolicy)
    def make_optimizer(self):
        config = self.config["torch_optimizer"]
        components = "model actor critics".split()
        if self.config["true_model"]:
            components = components[1:]

        optims = {k: ptu.build_optimizer(self.module[k], config[k]) for k in components}
        return collections.namedtuple("OptimizerCollection", components)(**optims)

    def set_transition_kernel(self, transition_kernel):
        """Use an external transition kernel to sample imaginary states."""
        torch_script = self.config["module"]["torch_script"]
        transition = EnvTransition(
            self.observation_space,
            self.action_space,
            transition_kernel,
            torch_script=torch_script,
        )
        self.transition = torch.jit.script(transition) if torch_script else transition
        self.check_model(self.transition)

    def check_model(self, sampler):
        """Verify that the transition model is appropriate for the desired estimator."""
        if self.config["grad_estimator"] == "SF":
            obs = self.convert_to_tensor(self.observation_space.sample())[None]
            act = self.convert_to_tensor(self.action_space.sample())[None]
            sample, logp = sampler(obs, act.requires_grad_())
            assert sample.grad_fn is None
            assert logp is not None
            logp.mean().backward()
            assert (
                act.grad is not None
            ), "Transition grad log_prob must exist for SF estimator"
            assert not torch.allclose(act.grad, torch.zeros_like(act))
        if self.config["grad_estimator"] == "PD":
            obs = self.convert_to_tensor(self.observation_space.sample())[None]
            act = self.convert_to_tensor(self.action_space.sample())[None]
            samp, _ = sampler(obs.requires_grad_(), act.requires_grad_())
            samp.mean().backward()
            assert (
                obs.grad is not None and act.grad is not None
            ), "Transition grad w.r.t. state and action must exist for PD estimator"
            assert not torch.allclose(obs.grad, torch.zeros_like(obs))
            assert not torch.allclose(act.grad, torch.zeros_like(act))

    @override(raypi.TorchPolicy)
    def learn_on_batch(self, samples):
        batch_tensors = self._lazy_tensor_dict(samples)

        info = {}
        info.update(self._update_critic(batch_tensors, self.module, self.config))
        if not self.config["true_model"]:
            info.update(self._update_model(batch_tensors, self.module, self.config))
        info.update(self._update_actor(batch_tensors, self.module, self.config))

        self.update_targets("critics", "target_critics")
        return self._learner_stats(info)

    def _update_critic(self, batch_tensors, module, config):
        with self.optimizer.critics.optimize():
            critic_loss, info = self.critic_loss(batch_tensors, module, config)
            critic_loss.backward()

        grad_stats = {
            "grad_norm(critic)": nn.utils.clip_grad_norm_(
                module.critics.parameters(), float("inf")
            ).item()
        }
        info.update(grad_stats)
        return info

    def critic_loss(self, batch_tensors, module, config):
        """Compute loss for Q value function."""
        obs = batch_tensors[SampleBatch.CUR_OBS]
        actions = batch_tensors[SampleBatch.ACTIONS]

        with torch.no_grad():
            target_values = self.critic_targets(batch_tensors, module, config)
        loss_fn = nn.MSELoss()
        values = torch.cat([m(obs, actions) for m in module.critics], dim=-1)
        critic_loss = loss_fn(values, target_values.unsqueeze(-1).expand_as(values))

        stats = {
            "q_mean": values.mean().item(),
            "q_max": values.max().item(),
            "q_min": values.min().item(),
            "loss(critic)": critic_loss.item(),
        }
        return critic_loss, stats

    def critic_targets(self, batch_tensors, module, config):
        """
        Compute 1-step approximation of Q^{\\pi}(s, a) for Clipped Double Q-Learning
        using target networks and batch transitions.
        """
        rewards = batch_tensors[SampleBatch.REWARDS]
        gamma = config["gamma"]
        next_obs = batch_tensors[SampleBatch.NEXT_OBS]
        dones = batch_tensors[SampleBatch.DONES]

        return rewards + gamma * self._clipped_target_value(next_obs, dones, module)

    @staticmethod
    def _clipped_target_value(obs, dones, module):
        acts = module.target_actor(obs)
        vals, _ = torch.cat([m(obs, acts) for m in module.target_critics], dim=-1).min(
            dim=-1
        )
        return torch.where(dones, torch.zeros_like(vals), vals)

    def _update_model(self, batch_tensors, module, config):
        with self.optimizer.model.optimize():
            if config["model_loss"] == "DAML":
                model_loss, info = self.daml_loss(batch_tensors, module, config)
            elif config["model_loss"] == "MLE":
                model_loss, info = self.mle_loss(batch_tensors, module)
            model_loss.backward()

        grad_stats = {
            "grad_norm(model)": nn.utils.clip_grad_norm_(
                module.model.parameters(), float("inf")
            ).item()
        }
        info.update(grad_stats)
        return info

    def daml_loss(self, batch_tensors, module, config):
        """Compute policy gradient-aware (PGA) model loss."""
        obs = batch_tensors[SampleBatch.CUR_OBS]
        actions = module.actor(obs).detach().requires_grad_()

        predictions = self.one_step_action_value_surrogate(obs, actions, module, config)
        targets = self.zero_step_action_values(obs, actions, module)

        temporal_diff_loss = torch.sum(predictions - targets)
        temporal_diff_loss.backward(create_graph=True)

        action_gradients = actions.grad
        # WARNING: may be ill-conditioned depending on the torch.norm() implementation
        daml_loss = torch.norm(action_gradients, p=2, dim=-1).mean()
        return daml_loss, {"loss(daml)": daml_loss.item()}

    def one_step_action_value_surrogate(self, obs, actions, module, config):
        """
        Compute 1-step approximation of Q^{\\pi}(s, a) for Deterministic Policy Gradient
        using target networks and model transitions.
        """
        gamma = config["gamma"]
        transition = (
            self.transition
            if config["true_model"]
            else module.model.sample
            if config["grad_estimator"] == "SF"
            else module.model.rsample
        )

        sample_shape = (config["num_model_samples"],)
        obs = obs.expand(sample_shape + obs.shape)
        actions = actions.expand(sample_shape + actions.shape)

        next_obs, logp = transition(obs, actions)
        rewards = self.reward(obs, actions, next_obs)
        # Assume virtual transition is not final
        dones = torch.ones_like(rewards).bool()
        next_values = self._clipped_target_value(obs, dones, self.module)
        values = rewards + gamma * next_values

        if config["grad_estimator"] == "SF":
            surrogate = torch.mean(logp * values.detach(), dim=0)
        elif config["grad_estimator"] == "PD":
            surrogate = torch.mean(values, dim=0)
        return surrogate

    @staticmethod
    def zero_step_action_values(obs, actions, module):
        """Compute Q^{\\pi}(s, a) directly using approximate critic."""
        action_values, _ = torch.cat(
            [m(obs, actions) for m in module.critics], dim=-1
        ).min(dim=-1)
        return action_values

    @staticmethod
    def mle_loss(batch_tensors, module):
        """Compute Maximum Likelihood Estimation (MLE) model loss."""
        avg_logp = module.model.log_prob(
            batch_tensors[SampleBatch.CUR_OBS],
            batch_tensors[SampleBatch.ACTIONS],
            batch_tensors[SampleBatch.NEXT_OBS],
        ).mean()
        loss = avg_logp.neg()
        return loss, {"loss(mle)": loss.item()}

    def _update_actor(self, batch_tensors, module, config):
        with self.optimizer.actor.optimize():
            policy_loss, info = self.madpg_loss(batch_tensors, module, config)
            policy_loss.backward()

        info.update(self.extra_policy_grad_info())
        return info

    def dpg_loss(self, batch_tensors, module):
        """Compute loss for deterministic policy gradient."""
        obs = batch_tensors[SampleBatch.CUR_OBS]

        actions = module.actor(obs)
        action_values = self.zero_step_action_values(obs, actions, module)
        max_objective = torch.mean(action_values)

        stats = {
            "loss(actor)": -max_objective.item(),
        }
        return -max_objective, stats

    def madpg_loss(self, batch_tensors, module, config):
        """Compute loss for model-aware deterministic policy gradient."""
        obs = batch_tensors[SampleBatch.CUR_OBS]

        actions = module.actor(obs)
        action_values = self.one_step_action_value_surrogate(
            obs, actions, module, config
        )
        max_objective = torch.mean(action_values)

        stats = {
            "loss(actor)": -max_objective.item(),
        }
        return -max_objective, stats

    def extra_policy_grad_info(self):
        """Return dict of extra info on policy gradient."""
        return {
            "grad_norm(actor)": nn.utils.clip_grad_norm_(
                self.module.actor.parameters(), float("inf")
            ).item()
        }


class EnvTransition(nn.Module):
    """Wrapper module around existing env transition function."""

    def __init__(self, obs_space, action_space, transition_kernel, torch_script=False):
        super().__init__()
        if torch_script:
            obs = torch.as_tensor(obs_space.sample())[None]
            action = torch.as_tensor(action_space.sample())[None]
            transition_kernel = torch.jit.trace(transition_kernel, (obs, action))
        self.transition_kernel = transition_kernel

    @override(nn.Module)
    def forward(self, obs, action):  # pylint:disable=arguments-differ
        return self.transition_kernel(obs, action)
