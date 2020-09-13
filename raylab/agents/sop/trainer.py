"""Trainer and configuration for SOP."""
from raylab.agents.off_policy import OffPolicyTrainer
from raylab.options import configure
from raylab.options import option

from .policy import SOPTorchPolicy


def sop_config(cls: type) -> type:
    """Add configurations for Streamlined Off-Policy-based agents."""
    for config_setter in [
        option(
            "dpg_loss",
            "default",
            help="""
            Type of Deterministic Policy Gradient to use.

            'default' backpropagates Q-value gradients through the critic network.

            'acme' uses Acme's implementation which recovers DPG via a MSE loss between
            the actor's action and the action + Q-value gradient. Allows monitoring the
            magnitude of the action-value gradient.""",
        ),
        option(
            "dqda_clipping",
            None,
            help="""
            Optional value by which to clip the action gradients. Only used with
            dpg_loss='acme'.""",
        ),
        option(
            "clip_dqda_norm",
            False,
            help="""
            Whether to clip action grads by norm or value. Only used with
            dpg_loss='acme'.""",
        ),
        option("torch_optimizer/actor", {"type": "Adam", "lr": 1e-3}),
        option("torch_optimizer/critics", {"type": "Adam", "lr": 1e-3}),
        option(
            "polyak",
            0.995,
            help="Interpolation factor in polyak averaging for target networks.",
        ),
        option(
            "policy_delay",
            1,
            help="Update policy every this number of calls to `learn_on_batch`",
        ),
    ]:
        cls = config_setter(cls)

    return cls


@configure
@sop_config
@option("module/type", "DDPG")
@option("module/actor/separate_behavior", True)
@option(
    "exploration_config/type", "raylab.utils.exploration.ParameterNoise", override=True
)
@option(
    "exploration_config/param_noise_spec",
    {"initial_stddev": 0.1, "desired_action_stddev": 0.2, "adaptation_coeff": 1.01},
    help="Options for parameter noise exploration",
)
@option("exploration_config/pure_exploration_steps", 1000)
@option("evaluation_config/explore", False, override=True)
class SOPTrainer(OffPolicyTrainer):
    """Single agent trainer for the Streamlined Off-Policy algorithm."""

    _name = "SOP"
    _policy = SOPTorchPolicy
