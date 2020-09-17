"""SOP policy class using PyTorch."""
import torch
import torch.nn as nn
from ray.rllib.utils import override

from raylab.options import configure
from raylab.options import option
from raylab.policy import TorchPolicy
from raylab.policy.action_dist import WrapDeterministicPolicy
from raylab.policy.losses import ActionDPG
from raylab.policy.losses import DeterministicPolicyGradient
from raylab.policy.losses import FittedQLearning
from raylab.policy.modules.critic import HardValue
from raylab.policy.off_policy import off_policy_options
from raylab.policy.off_policy import OffPolicyMixin
from raylab.torch.nn.utils import update_polyak
from raylab.torch.optim import build_optimizer
from raylab.utils.annotations import TensorDict


@configure
@off_policy_options
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
@option("torch_optimizer/actor", {"type": "Adam", "lr": 1e-3})
@option("torch_optimizer/critics", {"type": "Adam", "lr": 1e-3})
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
@option("module/type", "DDPG")
@option("module/actor/separate_behavior", True)
@option("exploration_config/type", "raylab.utils.exploration.ParameterNoise")
@option(
    "exploration_config/param_noise_spec",
    {"initial_stddev": 0.1, "desired_action_stddev": 0.2, "adaptation_coeff": 1.01},
    help="Options for parameter noise exploration",
)
@option("exploration_config/pure_exploration_steps", 1000)
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
        config = self.config["torch_optimizer"]
        components = "actor critics".split()

        mapping = {
            name: build_optimizer(getattr(self.module, name), config[name])
            for name in components
        }

        optimizers.update(mapping)
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
        return {
            f"grad_norm({component})": nn.utils.clip_grad_norm_(
                getattr(self.module, component).parameters(), float("inf")
            ).item()
        }

    @override(TorchPolicy)
    def get_weights(self):
        weights = super().get_weights()
        weights["grad_step"] = self._grad_step
        return weights

    @override(TorchPolicy)
    def set_weights(self, weights):
        self._grad_step = weights["grad_step"]
        super().set_weights(weights)
