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
        self.check_model(
            module.model.rsample
            if config["grad_estimator"] == "PD"
            else module.model.sample
        )
        module.model.zero_grad()
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
        obs = torch.randn(self.observation_space.shape)[None]
        act = torch.randn(self.action_space.shape)[None]
        if self.config["grad_estimator"] == "SF":
            sample, logp = sampler(obs, act.requires_grad_())
            assert sample.grad_fn is None
            assert logp is not None
            logp.mean().backward()
            assert (
                act.grad is not None
            ), "Transition grad log_prob must exist for SF estimator"
            assert not torch.allclose(act.grad, torch.zeros_like(act))
        if self.config["grad_estimator"] == "PD":
            sample, _ = sampler(obs.requires_grad_(), act.requires_grad_())
            sample.mean().backward()
            assert (
                act.grad is not None
            ), "Transition grad w.r.t. state and action must exist for PD estimator"
            assert not torch.allclose(act.grad, torch.zeros_like(act))

    @override(raypi.TorchPolicy)
    def learn_on_batch(self, samples):
        batch_tensors = self._lazy_tensor_dict(samples)

        info = {}
        info.update(self._update_critic(batch_tensors))
        if not self.config["true_model"]:
            info.update(self._update_model(batch_tensors))
        info.update(self._update_actor(batch_tensors))

        self.update_targets("critics", "target_critics")
        return self._learner_stats(info)

    def learn_critic(self, samples):
        """Update critics with samples."""
        batch_tensors = self._lazy_tensor_dict(samples)
        info = {}
        info.update(self._update_critic(batch_tensors))
        self.update_targets("critics", "target_critics")
        return self._learner_stats(info)

    def learn_model(self, samples):
        """Update model with samples."""
        batch_tensors = self._lazy_tensor_dict(samples)
        info = {}
        info.update(self._update_model(batch_tensors))
        return self._learner_stats(info)

    def learn_actor(self, samples):
        """Update actor with samples."""
        batch_tensors = self._lazy_tensor_dict(samples)
        info = {}
        info.update(self._update_actor(batch_tensors))
        return self._learner_stats(info)

    def _update_critic(self, batch_tensors):
        with self.optimizer.critics.optimize():
            critic_loss, info = self.critic_loss(batch_tensors)
            critic_loss.backward()

        info.update(self.extra_grad_info("critics"))
        return info

    def critic_loss(self, batch_tensors):
        """Compute loss for Q value function."""
        critics = self.module.critics
        obs = batch_tensors[SampleBatch.CUR_OBS]
        actions = batch_tensors[SampleBatch.ACTIONS]

        with torch.no_grad():
            target_values = self.critic_targets(batch_tensors)
        loss_fn = nn.MSELoss()
        values = torch.cat([m(obs, actions) for m in critics], dim=-1)
        critic_loss = loss_fn(values, target_values.unsqueeze(-1).expand_as(values))

        stats = {
            "q_mean": values.mean().item(),
            "q_max": values.max().item(),
            "q_min": values.min().item(),
            "loss(critic)": critic_loss.item(),
        }
        return critic_loss, stats

    def critic_targets(self, batch_tensors):
        """
        Compute 1-step approximation of Q^{\\pi}(s, a) for Clipped Double Q-Learning
        using target networks and batch transitions.
        """
        target_actor = self.module.target_actor
        target_critics = self.module.target_critics

        rewards = batch_tensors[SampleBatch.REWARDS]
        gamma = self.config["gamma"]
        next_obs = batch_tensors[SampleBatch.NEXT_OBS]
        dones = batch_tensors[SampleBatch.DONES]

        next_acts = target_actor(next_obs)
        target_values = self.clipped_value(next_obs, next_acts, dones, target_critics)
        return rewards + gamma * target_values

    @staticmethod
    def clipped_value(obs, acts, dones, critics):
        """Compute clipped Q^{\\pi}(s, a)."""
        vals, _ = torch.cat([m(obs, acts) for m in critics], dim=-1).min(dim=-1)
        return torch.where(dones, torch.zeros_like(vals), vals)

    def _update_model(self, batch_tensors):
        with self.optimizer.model.optimize():
            mle_loss, info = self.mle_loss(batch_tensors)

            if self.config["model_loss"] == "DAML":
                daml_loss, daml_info = self.daml_loss(batch_tensors)
                info.update(daml_info)

                alpha = self.config["mle_interpolation"]
                model_loss = alpha * mle_loss + (1 - alpha) * daml_loss
            else:
                model_loss = mle_loss

            model_loss.backward()

        info.update(self.extra_grad_info("model"))
        return info

    def daml_loss(self, batch_tensors):
        """Compute policy gradient-aware (PGA) model loss."""
        obs = batch_tensors[SampleBatch.CUR_OBS]
        actions = self.module.actor(obs).detach().requires_grad_()

        predictions = self.one_step_action_value_surrogate(obs, actions)
        targets = self.zero_step_action_values(obs, actions)

        temporal_diff = torch.sum(targets - predictions)
        (action_gradients,) = torch.autograd.grad(
            temporal_diff, actions, create_graph=True
        )

        daml_loss = torch.sum(action_gradients * action_gradients, dim=-1).mean()
        return (
            daml_loss,
            {"loss(action)": temporal_diff.item(), "loss(daml)": daml_loss.item()},
        )

    def one_step_action_value_surrogate(self, obs, actions, model_samples=1):
        """
        Compute 1-step approximation of Q^{\\pi}(s, a) for Deterministic Policy Gradient
        using target networks and model transitions.
        """
        actor = self.module.actor
        critics = self.module.critics
        sampler = (
            self.transition
            if self.config["true_model"]
            else self.module.model.sample
            if self.config["grad_estimator"] == "SF"
            else self.module.model.rsample
        )

        next_obs, rewards, dones, logp = self._generate_transition(
            obs, actions, self.reward, sampler, model_samples
        )
        # Next action grads shouldn't propagate
        with torch.no_grad():
            next_acts = actor(next_obs)
        next_values = self.clipped_value(next_obs, next_acts, dones, critics)
        values = rewards + self.config["gamma"] * next_values

        if self.config["grad_estimator"] == "SF":
            surrogate = torch.mean(logp * values.detach(), dim=0)
        elif self.config["grad_estimator"] == "PD":
            surrogate = torch.mean(values, dim=0)
        return surrogate

    @staticmethod
    def _generate_transition(obs, actions, reward_fn, sampler, num_samples):
        """Compute virtual transition as in env.step, with info replaced by logp."""
        sample_shape = (num_samples,)
        obs = obs.expand(sample_shape + obs.shape)
        actions = actions.expand(sample_shape + actions.shape)

        next_obs, logp = sampler(obs, actions)
        rewards = reward_fn(obs, actions, next_obs)
        # Assume virtual transition is not final
        dones = torch.zeros_like(rewards).bool()
        return next_obs, rewards, dones, logp

    def zero_step_action_values(self, obs, actions):
        """Compute Q^{\\pi}(s, a) directly using approximate critic."""
        dones = torch.zeros(obs.shape[:-1]).bool()
        return self.clipped_value(obs, actions, dones, self.module.critics)

    def mle_loss(self, batch_tensors):
        """Compute Maximum Likelihood Estimation (MLE) model loss."""
        avg_logp = self.module.model.log_prob(
            batch_tensors[SampleBatch.CUR_OBS],
            batch_tensors[SampleBatch.ACTIONS],
            batch_tensors[SampleBatch.NEXT_OBS],
        ).mean()
        loss = avg_logp.neg()
        return loss, {"loss(mle)": loss.item()}

    def _update_actor(self, batch_tensors):
        with self.optimizer.actor.optimize():
            policy_loss, info = self.madpg_loss(batch_tensors)
            policy_loss.backward()

        info.update(self.extra_grad_info("actor"))
        return info

    def dpg_loss(self, batch_tensors):
        """Compute loss for deterministic policy gradient."""
        obs = batch_tensors[SampleBatch.CUR_OBS]

        actions = self.module.actor(obs)
        action_values = self.zero_step_action_values(obs, actions)
        max_objective = torch.mean(action_values)
        loss = -max_objective

        stats = {"loss(actor)": loss.item()}
        return loss, stats

    def madpg_loss(self, batch_tensors):
        """Compute loss for model-aware deterministic policy gradient."""
        obs = batch_tensors[SampleBatch.CUR_OBS]

        actions = self.module.actor(obs)
        action_values = self.one_step_action_value_surrogate(
            obs, actions, self.config["num_model_samples"]
        )
        max_objective = torch.mean(action_values)
        loss = -max_objective

        stats = {"loss(actor)": loss.item()}
        return loss, stats

    @torch.no_grad()
    def extra_grad_info(self, component):
        """Clip grad norm and return statistics for component."""
        return {
            f"grad_norm({component})": nn.utils.clip_grad_norm_(
                self.module[component].parameters(),
                self.config["max_grad_norm"][component],
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
