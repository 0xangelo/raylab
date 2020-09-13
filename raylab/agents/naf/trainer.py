"""Continuous Q-Learning with Normalized Advantage Functions."""
from raylab.agents.off_policy import OffPolicyTrainer
from raylab.options import configure
from raylab.options import option

from .policy import NAFTorchPolicy


@configure
@option("torch_optimizer/type", "Adam")
@option("torch_optimizer/lr", 3e-4)
@option(
    "polyak",
    0.995,
    help="Interpolation factor in polyak averaging for target networks.",
)
@option("module/type", "NAF")
@option("module/separate_behavior", True)
@option(
    "exploration_config/type", "raylab.utils.exploration.ParameterNoise", override=True
)
@option(
    "exploration_config/param_noise_spec",
    {"initial_stddev": 0.1, "desired_action_stddev": 0.2, "adaptation_coeff": 1.01},
)
@option("exploration_config/pure_exploration_steps", 1000)
@option("evaluation_config/explore", False, override=True)
class NAFTrainer(OffPolicyTrainer):
    """Single agent trainer for NAF."""

    _name = "NAF"
    _policy = NAFTorchPolicy
