"""Policy for MAPO using PyTorch."""
import torch
import torch.nn as nn
from ray.rllib.utils import override

from raylab.losses import ClippedDoubleQLearning
from raylab.losses import DPGAwareModelLearning
from raylab.losses import MaximumLikelihood
from raylab.losses import ModelAwareDPG
from raylab.policy import EnvFnMixin
from raylab.policy import TargetNetworksMixin
from raylab.policy import TorchPolicy
from raylab.pytorch.optim import build_optimizer


class MAPOTorchPolicy(EnvFnMixin, TargetNetworksMixin, TorchPolicy):
    """Model-Aware Policy Optimization policy in PyTorch to use with RLlib."""

    # pylint: disable=abstract-method

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)

        self.loss_daml = DPGAwareModelLearning(
            self.module.actor,
            self.module.critics,
            gamma=self.config["gamma"],
            grad_estimator=self.config["grad_estimator"],
        )
        transition = (
            self.module.model.rsample
            if self.config["grad_estimator"] == "PD"
            else self.module.model.sample
        )
        self.loss_daml.set_model(transition)
        self.loss_mle = MaximumLikelihood(self.module.model)

        self.loss_actor = ModelAwareDPG(
            self.module.actor,
            self.module.critics,
            gamma=self.config["gamma"],
            num_model_samples=self.config["num_model_samples"],
            grad_estimator=self.config["grad_estimator"],
        )
        if not self.config["true_model"]:
            self.setup_madpg(transition)
            self.module.model.zero_grad()

        self.loss_critic = ClippedDoubleQLearning(
            self.module.critics,
            self.module.target_critics,
            self.module.target_actor,
            gamma=self.config["gamma"],
        )

    @override(EnvFnMixin)
    def set_reward_from_config(self, env_name: str, env_config: dict):
        super().set_reward_from_config(env_name, env_config)
        self.loss_daml.set_reward_fn(self.reward_fn)
        self.loss_actor.set_reward_fn(self.reward_fn)

    @override(EnvFnMixin)
    def set_dynamics_from_callable(self, function):
        super().set_dynamics_from_callable(function)
        self.setup_madpg(self.dynamics_fn)

    @staticmethod
    @override(TorchPolicy)
    def get_default_config():
        """Return the default configuration for MAPO."""
        # pylint: disable=cyclic-import
        from raylab.agents.mapo import DEFAULT_CONFIG

        return DEFAULT_CONFIG

    @override(TorchPolicy)
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
        return super().make_module(obs_space, action_space, config)

    @override(TorchPolicy)
    def make_optimizers(self):
        config = self.config["torch_optimizer"]
        components = "model actor critics".split()
        if self.config["true_model"]:
            components = components[1:]

        return {
            name: build_optimizer(self.module[name], config[name])
            for name in components
        }

    @override(TorchPolicy)
    def compile(self):
        super().compile()
        if self.dynamics_fn:
            obs = torch.randn(self.observation_space.shape)[None]
            act = torch.randn(self.action_space.shape)[None]
            self.dynamics_fn = torch.jit.trace(self.dynamics_fn, (obs, act))
            self.loss_actor.set_model(self.dynamics_fn)

    def setup_madpg(self, model):
        """Verify and use model for Model-Aware DPG."""
        obs = torch.randn(self.observation_space.shape)[None]
        act = torch.randn(self.action_space.shape)[None]
        if self.config["grad_estimator"] == "SF":
            sample, logp = model(obs, act.requires_grad_())
            assert sample.grad_fn is None
            assert logp is not None
            logp.mean().backward()
            assert (
                act.grad is not None
            ), "Transition grad log_prob must exist for SF estimator"
            assert not torch.allclose(act.grad, torch.zeros_like(act))
        if self.config["grad_estimator"] == "PD":
            sample, _ = model(obs.requires_grad_(), act.requires_grad_())
            sample.mean().backward()
            assert (
                act.grad is not None
            ), "Transition grad w.r.t. state and action must exist for PD estimator"
            assert not torch.allclose(act.grad, torch.zeros_like(act))

        self.loss_actor.set_model(model)

    @override(TorchPolicy)
    def learn_on_batch(self, samples):
        batch_tensors = self.lazy_tensor_dict(samples)

        info = {}
        info.update(self._update_critic(batch_tensors))
        if not self.config["true_model"]:
            info.update(self._update_model(batch_tensors))
        info.update(self._update_actor(batch_tensors))

        self.update_targets("critics", "target_critics")
        return info

    def _update_critic(self, batch_tensors):
        with self.optimizers.optimize("critics"):
            critic_loss, info = self.loss_critic(batch_tensors)
            critic_loss.backward()

        info.update(self.extra_grad_info("critics"))
        return info

    def _update_model(self, batch_tensors):
        with self.optimizers.optimize("model"):
            mle_loss, info = self.loss_mle(batch_tensors)
            info["loss(mle)"] = mle_loss.item()

            if self.config["model_loss"] == "DAML":
                daml_loss, daml_info = self.loss_daml(batch_tensors)
                info.update(daml_info)
                info["loss(daml)"] = daml_loss.item()

                alpha = self.config["mle_interpolation"]
                model_loss = alpha * mle_loss + (1 - alpha) * daml_loss
            else:
                model_loss = mle_loss

            model_loss.backward()
            info["loss(model)"] = model_loss.item()

        info.update(self.extra_grad_info("model"))
        return info

    def _update_actor(self, batch_tensors):
        with self.optimizers.optimize("actor"):
            policy_loss, info = self.loss_actor(batch_tensors)
            policy_loss.backward()

        info.update(self.extra_grad_info("actor"))
        return info

    @torch.no_grad()
    def extra_grad_info(self, component):
        """Clip grad norm and return statistics for component."""
        return {
            f"grad_norm({component})": nn.utils.clip_grad_norm_(
                self.module[component].parameters(),
                self.config["max_grad_norm"][component],
            ).item()
        }
