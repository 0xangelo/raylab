"""Trainer and configuration for SOP."""
from raylab.agents import trainer
from raylab.agents.off_policy import OffPolicyTrainer

from .policy import SOPTorchPolicy


DEFAULT_MODULE = {
    "type": "DDPG",
    "actor": {"separate_behavior": True},
    "critic": {"double_q": True},
}


@trainer.config("torch_optimizer/actor", {"type": "Adam", "lr": 1e-3})
@trainer.config("torch_optimizer/critics", {"type": "Adam", "lr": 1e-3})
@trainer.config(
    "polyak",
    0.995,
    info="Interpolation factor in polyak averaging for target networks.",
)
@trainer.config(
    "policy_delay",
    1,
    info="Update policy every this number of calls to `learn_on_batch`",
)
@trainer.config("module", DEFAULT_MODULE, override=True)
@trainer.config(
    "exploration_config/type", "raylab.utils.exploration.ParameterNoise", override=True
)
@trainer.config(
    "exploration_config/param_noise_spec",
    {"initial_stddev": 0.1, "desired_action_stddev": 0.2, "adaptation_coeff": 1.01},
    info="Options for parameter noise exploration",
)
@trainer.config("exploration_config/pure_exploration_steps", 1000)
@trainer.config("evaluation_config/explore", False, override=True)
@OffPolicyTrainer.with_base_specs
class SOPTrainer(OffPolicyTrainer):
    """Single agent trainer for Streamlined Off-Policy Algorithm."""

    _name = "SOP"
    _policy = SOPTorchPolicy
