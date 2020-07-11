"""SAC policy class using PyTorch."""
import torch
import torch.nn as nn
from ray.rllib.utils import override

from raylab.policy import TorchPolicy
from raylab.policy.action_dist import WrapStochasticPolicy
from raylab.policy.losses import MaximumEntropyDual
from raylab.policy.losses import ReparameterizedSoftPG
from raylab.policy.losses import SoftCDQLearning
from raylab.pytorch.nn.utils import update_polyak
from raylab.pytorch.optim import build_optimizer


class SACTorchPolicy(TorchPolicy):
    """Soft Actor-Critic policy in PyTorch to use with RLlib."""

    # pylint:disable=abstract-method
    dist_class = WrapStochasticPolicy

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self.loss_actor = ReparameterizedSoftPG(self.module.actor, self.module.critics)
        self.loss_critic = SoftCDQLearning(
            self.module.critics, self.module.target_critics, self.module.actor
        )
        self.loss_critic.gamma = self.config["gamma"]

        target_entropy = (
            -action_space.shape[0]
            if self.config["target_entropy"] == "auto"
            else self.config["target_entropy"]
        )
        self.loss_alpha = MaximumEntropyDual(
            self.module.alpha, self.module.actor.sample, target_entropy
        )

    @staticmethod
    @override(TorchPolicy)
    def get_default_config():
        """Return the default config for SAC."""
        # pylint:disable=cyclic-import,protected-access
        from raylab.agents.sac import SACTrainer

        return SACTrainer._default_config

    @override(TorchPolicy)
    def make_optimizers(self):
        config = self.config["torch_optimizer"]
        components = "actor critics alpha".split()

        return {
            name: build_optimizer(getattr(self.module, name), config[name])
            for name in components
        }

    @override(TorchPolicy)
    def learn_on_batch(self, samples):
        batch_tensors = self.lazy_tensor_dict(samples)
        info = {}

        alpha = self.module.alpha().item()
        self.loss_critic.alpha = alpha
        self.loss_actor.alpha = alpha

        info.update(self._update_critic(batch_tensors))
        info.update(self._update_actor(batch_tensors))
        if self.config["target_entropy"] is not None:
            info.update(self._update_alpha(batch_tensors))

        update_polyak(
            self.module.critics, self.module.target_critics, self.config["polyak"]
        )
        return info

    def _update_critic(self, batch_tensors):
        with self.optimizers.optimize("critics"):
            critic_loss, info = self.loss_critic(batch_tensors)
            critic_loss.backward()

        info.update(self.extra_grad_info("critics"))
        return info

    def _update_actor(self, batch_tensors):
        with self.optimizers.optimize("actor"):
            actor_loss, info = self.loss_actor(batch_tensors)
            actor_loss.backward()

        info.update(self.extra_grad_info("actor"))
        return info

    def _update_alpha(self, batch_tensors):
        with self.optimizers.optimize("alpha"):
            alpha_loss, info = self.loss_alpha(batch_tensors)
            alpha_loss.backward()

        info.update(self.extra_grad_info("alpha"))
        return info

    @torch.no_grad()
    def extra_grad_info(self, component):
        """Return statistics right after components are updated."""
        return {
            f"grad_norm({component})": nn.utils.clip_grad_norm_(
                getattr(self.module, component).parameters(), float("inf")
            ).item()
        }
