"""SAC policy class using PyTorch."""
import torch
import torch.nn as nn
from ray.rllib.utils import override

from raylab.options import configure
from raylab.options import option
from raylab.policy import TorchPolicy
from raylab.policy.action_dist import WrapStochasticPolicy
from raylab.policy.losses import FittedQLearning
from raylab.policy.losses import MaximumEntropyDual
from raylab.policy.losses import ReparameterizedSoftPG
from raylab.policy.modules.critic import SoftValue
from raylab.torch.nn.utils import update_polyak
from raylab.torch.optim import build_optimizer


def sac_config(cls: type) -> type:
    """Add configurations for Soft Actor-Critic-based agents."""

    for config_setter in [
        option(
            "target_entropy",
            None,
            help="""Target entropy for temperature parameter optimization.

            If 'auto', will use the heuristic provided in the SAC paper,
            H = -dim(A), where A is the action space

            If 'tf-agents', will use the TFAgents implementation,
            H = -dim(A) / 2, where A is the action space
            """,
        ),
        option("torch_optimizer/actor", {"type": "Adam", "lr": 1e-3}),
        option("torch_optimizer/critics", {"type": "Adam", "lr": 1e-3}),
        option("torch_optimizer/alpha", {"type": "Adam", "lr": 1e-3}),
        option(
            "polyak",
            0.995,
            help="Interpolation factor in polyak averaging for target networks.",
        ),
        option("exploration_config/type", "raylab.utils.exploration.StochasticActor"),
    ]:
        cls = config_setter(cls)

    return cls


@configure
@sac_config
@option("module", {"type": "SAC", "critic": {"double_q": True}}, override=True)
@option("exploration_config/pure_exploration_steps", 1000)
class SACTorchPolicy(TorchPolicy):
    """Soft Actor-Critic policy in PyTorch to use with RLlib."""

    # pylint:disable=abstract-method
    dist_class = WrapStochasticPolicy

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)

        self._setup_actor_loss()
        self._setup_critic_loss()
        self._setup_alpha_loss()

    def _setup_actor_loss(self):
        self.loss_actor = ReparameterizedSoftPG(
            actor=self.module.actor, critic=self.module.critics, alpha=self.module.alpha
        )

    def _setup_critic_loss(self):
        module = self.module
        soft_target = SoftValue(module.actor, module.target_critics, module.alpha)
        self.loss_critic = FittedQLearning(module.critics, soft_target)
        self.loss_critic.gamma = self.config["gamma"]

    def _setup_alpha_loss(self):
        action_size = self.action_space.shape[0]
        if self.config["target_entropy"] == "auto":
            target_entropy = -action_size
        elif self.config["target_entropy"] == "tf-agents":
            target_entropy = -action_size / 2
        else:
            target_entropy = self.config["target_entropy"]

        self.loss_alpha = MaximumEntropyDual(
            self.module.alpha, self.module.actor.sample, target_entropy
        )

    @override(TorchPolicy)
    def _make_optimizers(self):
        optimizers = super()._make_optimizers()
        config = self.config["torch_optimizer"]

        components = "actor critics alpha".split()
        mapping = {
            name: build_optimizer(getattr(self.module, name), config[name])
            for name in components
        }

        optimizers.update(mapping)
        return optimizers

    @override(TorchPolicy)
    def learn_on_batch(self, samples):
        batch_tensors = self.lazy_tensor_dict(samples)
        info = {}

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
