"""SOP policy class using PyTorch."""
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
from raylab.policy.losses import ActionDPG, DeterministicPolicyGradient, FittedQLearning
from raylab.policy.off_policy import OffPolicyMixin


@configure
@option("buffer_size", int(1e6))
@option("batch_size", 256)
@option("std_obs", False)
@option("improvement_steps", 1)
@option(
    "dpg_loss",
    "default",
    help="""
    Type of Deterministic Policy Gradient to use.

    'default' backpropagates Q-value gradients through the critic network.

    'acme' uses Acme's implementation which recovers DPG via a MSE loss between
    the actor's action and the action + Q-value gradient. Allows monitoring the
    magnitude of the action-value gradient.""",
)
@option(
    "dqda_clipping",
    None,
    help="""
    Optional value by which to clip the action gradients. Only used with
    dpg_loss='acme'.""",
)
@option(
    "clip_dqda_norm",
    False,
    help="""
    Whether to clip action grads by norm or value. Only used with
    dpg_loss='acme'.""",
)
@option("optimizer/actor", {"type": "Adam", "lr": 3e-4})
@option("optimizer/critics", {"type": "Adam", "lr": 3e-4})
@option(
    "polyak",
    0.995,
    help="Interpolation factor in polyak averaging for target networks.",
)
@option(
    "policy_delay",
    1,
    help="Update policy every this number of calls to `learn_on_batch`",
)
@option("module/type", "SOP")
@option("exploration_config/type", "raylab.utils.exploration.GaussianNoise")
@option("exploration_config/noise_stddev", 0.3)
@option("exploration_config/pure_exploration_steps", 10000)
class SOPTorchPolicy(OffPolicyMixin, TorchPolicy):
    """Streamlined Off-Policy policy in PyTorch to use with RLlib."""

    # pylint:disable=abstract-method
    dist_class = WrapDeterministicPolicy

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._make_actor_loss()
        target_value = HardValue(self.module.target_actor, self.module.target_critics)
        self.loss_critic = FittedQLearning(self.module.critics, target_value)
        self.loss_critic.gamma = self.config["gamma"]
        self._grad_step = 0
        self._info = {}

        self.build_replay_buffer()

    def _make_actor_loss(self):
        if self.config["dpg_loss"] == "default":
            self.loss_actor = DeterministicPolicyGradient(
                self.module.actor,
                self.module.critics,
            )
        elif self.config["dpg_loss"] == "acme":
            self.loss_actor = ActionDPG(self.module.actor, self.module.critics)
            self.loss_actor.dqda_clipping = self.config["dqda_clipping"]
            self.loss_actor.clip_norm = self.config["clip_dqda_norm"]
        else:
            dpg_loss = self.config["dpg_loss"]
            raise ValueError(
                f"Invalid config for 'dpg_loss': {dpg_loss}."
                " Choose between 'default' and 'acme'"
            )

    @override(TorchPolicy)
    def _make_optimizers(self):
        optimizers = super()._make_optimizers()
        config = self.config["optimizer"]
        optimizers["actor"] = build_optimizer(self.module.actor, config["actor"])
        optimizers["critics"] = build_optimizer(self.module.critics, config["critics"])
        return optimizers

    @override(OffPolicyMixin)
    def improve_policy(self, batch: TensorDict):
        self._grad_step += 1
        self._info["grad_steps"] = self._grad_step

        self._info.update(self._update_critic(batch))
        if self._grad_step % self.config["policy_delay"] == 0:
            self._info.update(self._update_policy(batch))

        critics, target_critics = self.module.critics, self.module.target_critics
        update_polyak(critics, target_critics, self.config["polyak"])
        return self._info.copy()

    def _update_critic(self, batch_tensors):
        with self.optimizers.optimize("critics"):
            loss, info = self.loss_critic(batch_tensors)
            loss.backward()

        info.update(self.extra_grad_info("critics"))
        return info

    def _update_policy(self, batch_tensors):
        with self.optimizers.optimize("actor"):
            loss, info = self.loss_actor(batch_tensors)
            loss.backward()

        info.update(self.extra_grad_info("actor"))
        return info

    @torch.no_grad()
    def extra_grad_info(self, component):
        """Return statistics right after components are updated."""
        params = getattr(self.module, component).parameters()
        return {f"grad_norm({component})": clip_grad_norm_(params, float("inf")).item()}

    @override(TorchPolicy)
    def get_weights(self):
        weights = super().get_weights()
        weights["grad_step"] = self._grad_step
        return weights

    @override(TorchPolicy)
    def set_weights(self, weights):
        self._grad_step = weights["grad_step"]
        super().set_weights(weights)
