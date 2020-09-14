"""Generic Trainer and base configuration for model-based agents."""
import logging
from typing import Any
from typing import Callable

from ray.rllib import Policy
from ray.rllib.env.env_context import EnvContext
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.utils.types import EnvType
from ray.rllib.utils.types import PartialTrainerConfigDict

from raylab.utils.wandb import WandBLogger

from .trainer import Trainer


logger = logging.getLogger(__name__)


def set_policy_with_env_fn(worker_set: WorkerSet, fn_type: str = "reward"):
    """Set the desired environment function for all policies in the worker set.

    Args:
        worker_set: A worker set instance, usually from a trainer
        fn_type: The type of environment function, either 'reward',
            'termination', or 'dynamics'
        from_env: Whether to retrieve the function from the environment instance
            or from the global registry
    """
    worker_set.foreach_worker(
        lambda w: w.foreach_policy(
            lambda p, _: _set_from_env_if_possible(p, w.env, fn_type)
        )
    )


def _set_from_env_if_possible(policy: Policy, env: Any, fn_type: str = "reward"):
    env_fn = getattr(env, fn_type + "_fn", None)
    if fn_type == "reward":
        if env_fn:
            policy.set_reward_from_callable(env_fn)
        else:
            policy.set_reward_from_config()
    elif fn_type == "termination":
        if env_fn:
            policy.set_termination_from_callable(env_fn)
        else:
            policy.set_termination_from_config()
    elif fn_type == "dynamics":
        if env_fn:
            policy.set_dynamics_from_callable(env_fn)
        else:
            raise ValueError(
                f"Environment '{env}' has no '{fn_type + '_fn'}' attribute"
            )
    else:
        raise ValueError(f"Invalid env function type '{fn_type}'")


class ModelBasedTrainer(Trainer):
    """Generic trainer for model-based agents.

    Sets reward and termination functions for policies. These functions must be
    either:
    * Registered via `raylab.envs.register_reward_fn` and
      `raylab.envs.register_termination_fn`
    * Accessible attributes of the environment as `reward_fn` and
      `termination_fn`. These should not be bound instance methods; all
      necessary information should be encoded in the inputs, (state, action,
      and next state) i.e., the states should be markovian.
    """

    # pylint:disable=abstract-method
    _name: str

    def _init(
        self,
        config: PartialTrainerConfigDict,
        env_creator: Callable[[EnvContext], EnvType],
    ):
        self.config = config = self.restore_reserved()
        self.validate_config(config)
        self._policy = cls = self.get_policy_class(config)

        self.before_init()

        # Creating all workers (excluding evaluation workers).
        num_workers = config["num_workers"]
        self.workers = self._make_workers(env_creator, cls, config, num_workers)
        self.set_env_fns()
        self.optimize_policy_backend()
        self.train_exec_impl = self.execution_plan(self.workers, config)
        self.wandb = WandBLogger(config, self._name)

        self.after_init()

    def set_env_fns(self):
        """Set reward and termination functions for policies."""
        set_policy_with_env_fn(self.workers, fn_type="reward")
        set_policy_with_env_fn(self.workers, fn_type="termination")
