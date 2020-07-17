"""Trainer and configuration for SOP."""
from raylab.agents import trainer
from raylab.agents.off_policy import OffPolicyTrainer

from .policy import SOPTorchPolicy


def sop_config(cls: type) -> type:
    """Add configurations for Streamlined Off-Policy-based agents."""
    for config_setter in [
        trainer.option(
            "dpg_loss",
            "default",
            help="""
            Type of Deterministic Policy Gradient to use.

            'default' backpropagates Q-value gradients through the critic network.

            'acme' uses Acme's implementation which recovers DPG via a MSE loss between
            the actor's action and the action + Q-value gradient. Allows monitoring the
            magnitude of the action-value gradient.""",
        ),
        trainer.option(
            "dqda_clipping",
            None,
            help="""
            Optional value by which to clip the action gradients. Only used with
            dpg_loss='acme'.""",
        ),
        trainer.option(
            "clip_dqda_norm",
            False,
            help="""
            Whether to clip action grads by norm or value. Only used with
            dpg_loss='acme'.""",
        ),
        trainer.option("torch_optimizer/actor", {"type": "Adam", "lr": 1e-3}),
        trainer.option("torch_optimizer/critics", {"type": "Adam", "lr": 1e-3}),
        trainer.option(
            "polyak",
            0.995,
            help="Interpolation factor in polyak averaging for target networks.",
        ),
        trainer.option(
            "policy_delay",
            1,
            help="Update policy every this number of calls to `learn_on_batch`",
        ),
    ]:
        cls = config_setter(cls)

    return cls


@trainer.configure
@sop_config
@trainer.option("module/type", "DDPG")
@trainer.option("module/actor/separate_behavior", True)
@trainer.option(
    "exploration_config/type", "raylab.utils.exploration.ParameterNoise", override=True
)
@trainer.option(
    "exploration_config/param_noise_spec",
    {"initial_stddev": 0.1, "desired_action_stddev": 0.2, "adaptation_coeff": 1.01},
    help="Options for parameter noise exploration",
)
@trainer.option("exploration_config/pure_exploration_steps", 1000)
@trainer.option("evaluation_config/explore", False, override=True)
class SOPTrainer(OffPolicyTrainer):
    """Single agent trainer for the Streamlined Off-Policy algorithm."""

    _name = "SOP"
    _policy = SOPTorchPolicy
