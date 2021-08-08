"""NAF policy class using PyTorch."""
import torch
from nnrl.nn.critic import ClippedVValue
from nnrl.nn.utils import update_polyak
from nnrl.optim import build_optimizer
from nnrl.types import TensorDict
from ray.rllib.utils import override
from torch import nn

from raylab.options import configure, option
from raylab.policy import TorchPolicy
from raylab.policy.action_dist import WrapDeterministicPolicy
from raylab.policy.losses import FittedQLearning
from raylab.policy.off_policy import OffPolicyMixin, off_policy_options


@configure
@off_policy_options
@option("optimizer/type", "Adam")
@option("optimizer/lr", 3e-4)
@option(
    "polyak",
    0.995,
    help="Interpolation factor in polyak averaging for target networks.",
)
@option("module/type", "NAF")
@option("module/separate_behavior", True)
@option("exploration_config/type", "raylab.utils.exploration.ParameterNoise")
@option(
    "exploration_config/param_noise_spec",
    {"initial_stddev": 0.1, "desired_action_stddev": 0.2, "adaptation_coeff": 1.01},
)
@option("exploration_config/pure_exploration_steps", 1000)
class NAFTorchPolicy(OffPolicyMixin, TorchPolicy):
    """Normalized Advantage Function policy in Pytorch to use with RLlib."""

    # pylint:disable=abstract-method
    dist_class = WrapDeterministicPolicy

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = FittedQLearning(
            self.module.critics, ClippedVValue(self.module.target_vcritics)
        )
        self.loss_fn.gamma = self.config["gamma"]

        self.build_replay_buffer()

    @override(TorchPolicy)
    def _make_module(self, obs_space, action_space, config):
        module_config = config["module"]
        module_config["type"] = "NAF"
        # pylint:disable=no-member
        return super()._make_module(obs_space, action_space, config)

    @override(TorchPolicy)
    def _make_optimizers(self):
        optimizers = super()._make_optimizers()
        optimizers.update(
            naf=build_optimizer(self.module.critics, self.config["optimizer"])
        )
        return optimizers

    @override(OffPolicyMixin)
    def improve_policy(self, batch: TensorDict) -> dict:
        with self.optimizers.optimize("naf"):
            loss, info = self.loss_fn(batch)
            loss.backward()

        info.update(self.extra_grad_info())

        vcritics, target_vcritics = self.module.vcritics, self.module.target_vcritics
        update_polyak(vcritics, target_vcritics, self.config["polyak"])
        return info

    @torch.no_grad()
    def extra_grad_info(self):
        """Compute gradient norm for components."""
        return {
            "grad_norm": nn.utils.clip_grad_norm_(
                self.module.critics.parameters(), float("inf")
            ).item()
        }
