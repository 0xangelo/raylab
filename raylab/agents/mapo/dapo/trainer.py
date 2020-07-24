# pylint:disable=missing-module-docstring
import raylab.envs as envs
from raylab.agents import trainer
from raylab.agents.off_policy import OffPolicyTrainer
from raylab.agents.sac.trainer import sac_config

from .policy import DAPOTorchPolicy


@trainer.configure
@trainer.option("losses/", help="Configurations for actor loss function")
@trainer.option(
    "losses/grad_estimator",
    "PD",
    help="""Gradient estimator for optimizing expectations.

    Possible types include
    SF: score function
    PD: pathwise derivative
    """,
)
@trainer.option(
    "losses/model_samples",
    4,
    help="Number of next states to sample from the model when calculating the"
    " model-aware deterministic policy gradient",
)
@trainer.option("module/type", "SAC")
@sac_config
class DAPOTrainer(OffPolicyTrainer):
    """Single-agent trainer for Dynamics-Aware Policy Optimization"""

    _name = "DAPO"
    _policy = DAPOTorchPolicy

    def _init(self, config, env_creator):
        super()._init(config, env_creator)

        policy = self.get_policy()
        worker = self.workers.local_worker()

        if envs.has_reward_fn(config["env"]):
            policy.set_reward_from_config(config["env"], config["env_config"])
        else:
            policy.set_reward_from_callable(worker.env.reward_fn)

        if envs.has_termination_fn(config["env"]):
            policy.set_termination_from_config(config["env"], config["env_config"])
        else:
            policy.set_termination_from_callable(worker.env.termination_fn)

        policy.set_dynamics_from_callable(worker.env.transition_fn)
