"""Continuous Q-Learning with Normalized Advantage Functions."""
from raylab.agents import trainer
from raylab.agents.off_policy import OffPolicyTrainer

from .policy import NAFTorchPolicy


@trainer.configure
@trainer.option("torch_optimizer/type", "Adam")
@trainer.option("torch_optimizer/lr", 3e-4)
@trainer.option(
    "polyak",
    0.995,
    help="Interpolation factor in polyak averaging for target networks.",
)
@trainer.option("module/type", "NAF")
@trainer.option("module/separate_behavior", True)
@trainer.option(
    "exploration_config/type", "raylab.utils.exploration.ParameterNoise", override=True
)
@trainer.option(
    "exploration_config/param_noise_spec",
    {"initial_stddev": 0.1, "desired_action_stddev": 0.2, "adaptation_coeff": 1.01},
)
@trainer.option("exploration_config/pure_exploration_steps", 1000)
@trainer.option("evaluation_config/explore", False, override=True)
class NAFTrainer(OffPolicyTrainer):
    """Single agent trainer for NAF."""

    _name = "NAF"
    _policy = NAFTorchPolicy
