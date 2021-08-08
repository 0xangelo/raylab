"""TD3 policy class using PyTorch."""
import torch
from nnrl.nn.critic import HardValue
from nnrl.nn.utils import update_polyak
from nnrl.optim import build_optimizer
from nnrl.types import TensorDict
from ray.rllib.utils import override
from torch.nn.utils import clip_grad_norm_

from raylab.options import configure, option
from raylab.policy import TorchPolicy
from raylab.policy.action_dist import WrapDeterministicPolicy
from raylab.policy.losses import DeterministicPolicyGradient, FittedQLearning
from raylab.policy.off_policy import OffPolicyMixin, off_policy_options


@configure
@off_policy_options
@option(
    "policy_delay",
    2,
    help="Update policy every this number of calls to `learn_on_batch`",
)
@option(
    "polyak",
    0.995,
    help="Interpolation factor in polyak averaging for target networks.",
)
@option("module/type", "TD3")
@option("optimizer/actor", {"type": "Adam", "lr": 1e-3})
@option("optimizer/critics", {"type": "Adam", "lr": 1e-3})
@option("exploration_config/type", "raylab.utils.exploration.GaussianNoise")
@option("exploration_config/noise_stddev", 0.3)
class TD3TorchPolicy(OffPolicyMixin, TorchPolicy):
    """TD3 policy in Pytorch for RLlib."""

    dist_class = WrapDeterministicPolicy

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._make_actor_loss()
        self._make_critic_loss()
        self._grad_step = 0
        self._info = {}

        self.build_replay_buffer()

    def _make_actor_loss(self):
        self.loss_actor = DeterministicPolicyGradient(
            self.module.actor,
            self.module.critics,
        )

    def _make_critic_loss(self):
        target_value = HardValue(self.module.target_actor, self.module.target_critics)
        self.loss_critic = FittedQLearning(self.module.critics, target_value)
        self.loss_critic.gamma = self.config["gamma"]

    @override(TorchPolicy)
    def _make_optimizers(self):
        optimizers = super()._make_optimizers()
        config = self.config["optimizer"]
        optimizers["actor"] = build_optimizer(self.module.actor, config["actor"])
        optimizers["critics"] = build_optimizer(self.module.critics, config["critics"])
        return optimizers

    @override(OffPolicyMixin)
    def improve_policy(self, batch: TensorDict) -> dict:
        self._grad_step += 1
        self._info["grad_steps"] = self._grad_step

        self._info.update(self._update_critic(batch))
        if self._grad_step % self.config["policy_delay"] == 0:
            self._info.update(self._update_policy(batch))

        return self._info.copy()

    def _update_critic(self, batch: TensorDict) -> dict:
        with self.optimizers.optimize("critics"):
            loss, info = self.loss_critic(batch)
            loss.backward()
            info.update(self.extra_grad_info("critics"))

        main, target = self.module.critics, self.module.target_critics
        update_polyak(main, target, self.config["polyak"])
        return info

    def _update_policy(self, batch: TensorDict) -> dict:
        with self.optimizers.optimize("actor"):
            loss, info = self.loss_actor(batch)
            loss.backward()
            info.update(self.extra_grad_info("actor"))

        main, target = self.module.actor, self.module.target_actor
        update_polyak(main, target, self.config["polyak"])
        return info

    @torch.no_grad()
    def extra_grad_info(self, component: str) -> dict:
        """Return gradient statistics for the given component."""
        params = getattr(self.module, component).parameters()
        clip = float("inf")
        return {f"grad_norm({component})": clip_grad_norm_(params, clip).item()}

    @override(TorchPolicy)
    def get_weights(self) -> dict:
        weights = super().get_weights()
        weights["grad_step"] = self._grad_step
        return weights

    @override(TorchPolicy)
    def set_weights(self, weights: dict):
        self._grad_step = weights["grad_step"]
        super().set_weights(weights)
