"""Continuous Q-Learning with Normalized Advantage Functions."""
from raylab.agents import trainer
from raylab.agents.off_policy import OffPolicyTrainer

from .policy import NAFTorchPolicy


@trainer.config(
    "clipped_double_q", False, info="Whether to use Clipped Double Q-Learning"
)
@trainer.config("torch_optimizer/type", "Adam")
@trainer.config("torch_optimizer/lr", 3e-4)
@trainer.config(
    "polyak",
    0.995,
    info="Interpolation factor in polyak averaging for target networks.",
)
@trainer.config(
    "exploration_config/type", "raylab.utils.exploration.ParameterNoise", override=True
)
@trainer.config(
    "exploration_config/param_noise_spec",
    {"initial_stddev": 0.1, "desired_action_stddev": 0.2, "adaptation_coeff": 1.01},
)
@trainer.config("exploration_config/pure_exploration_steps", 1000)
@trainer.config("evaluation_config/explore", False, override=True)
@OffPolicyTrainer.with_base_specs
class NAFTrainer(OffPolicyTrainer):
    """Single agent trainer for NAF."""

    _name = "NAF"
    _policy = NAFTorchPolicy
