# pylint:disable=missing-module-docstring
from raylab.agents import trainer
from raylab.agents.model_based import set_policy_with_env_fn
from raylab.agents.off_policy import OffPolicyTrainer
from raylab.agents.sac.trainer import sac_config

from .policy import DAPOTorchPolicy


@trainer.configure
@trainer.option("losses/", help="Configurations for actor loss function")
@trainer.option(
    "losses/grad_estimator",
    default="PD",
    help="""Gradient estimator for optimizing expectations.

    Possible types include
    SF: score function
    PD: pathwise derivative
    """,
)
@trainer.option(
    "losses/model_samples",
    default=1,
    help="Number of next states to sample from the model when calculating the"
    " model-aware deterministic policy gradient",
)
@trainer.option("module/type", default="SAC")
@sac_config
class DAPOTrainer(OffPolicyTrainer):
    """Single-agent trainer for Dynamics-Aware Policy Optimization"""

    _name = "DAPO"
    _policy = DAPOTorchPolicy

    def _init(self, config, env_creator):
        super()._init(config, env_creator)
        set_policy_with_env_fn(self.workers, fn_type="reward")
        set_policy_with_env_fn(self.workers, fn_type="termination")
        set_policy_with_env_fn(self.workers, fn_type="dynamics")
