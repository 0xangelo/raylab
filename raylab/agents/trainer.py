# pylint:disable=missing-module-docstring
import logging
import time
from abc import ABCMeta
from typing import Callable, Iterable, Optional, Type

from ray.exceptions import RayError
from ray.rllib import Policy
from ray.rllib.agents import Trainer as RLlibTrainer
from ray.rllib.agents.trainer import MAX_WORKER_FAILURE_RETRIES
from ray.rllib.agents.trainer_template import default_execution_plan
from ray.rllib.env.env_context import EnvContext
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.utils import override as overrides
from ray.rllib.utils.typing import (
    EnvType,
    PartialTrainerConfigDict,
    ResultDict,
    TrainerConfigDict,
)
from ray.tune.resources import Resources
from ray.tune.trainable import Trainable

from raylab.options import TrainerOptions, configure, option

logger = logging.getLogger(__name__)


# ==============================================================================
# Streamlined Trainer
# ==============================================================================
@configure
@option(
    "policy/",
    allow_unknown_subkeys=True,
    help="""Sub-configurations for the policy class.""",
)
@option("framework", default="torch", override=True)
@option(
    "simple_optimizer",
    # RLlib attempts to use Tensorflow if this is true
    default=True,
    override=True,
)
class Trainer(RLlibTrainer, metaclass=ABCMeta):
    """Base class for raylab trainers."""

    config: TrainerConfigDict
    raw_user_config: PartialTrainerConfigDict
    env_creator: Callable[[EnvContext], EnvType]
    global_vars: dict
    workers: WorkerSet
    evaluation_workers: Optional[WorkerSet]
    train_exec_impl: Iterable[ResultDict]
    options: TrainerOptions = TrainerOptions()
    _name: str
    _policy_class: Type[Policy]
    _env_id: str
    _true_config: TrainerConfigDict

    def setup(self, config: PartialTrainerConfigDict):
        if self._env_id:
            config["env"] = self._env_id
        self._true_config = self.options.merge_defaults_with(config)
        super().setup(self.options.rllib_subconfig(self._true_config))

    @property
    def _default_config(self) -> TrainerConfigDict:
        return self.options.rllib_defaults

    def _init(
        self,
        config: PartialTrainerConfigDict,
        env_creator: Callable[[EnvContext], EnvType],
    ):
        self.config = config = self.restore_reserved()
        self.validate_config(config)
        self._policy_class = cls = self.get_policy_class()

        self.before_init()

        # Creating all workers (excluding evaluation workers).
        num_workers = config["num_workers"]
        self.workers = self._make_workers(
            env_creator=env_creator,
            validate_env=None,
            policy_class=cls,
            config=config,
            num_workers=num_workers,
        )
        self.train_exec_impl = self.execution_plan(self.workers, config)

        self.after_init()

        self.optimize_policy_backend()

    def restore_reserved(self) -> TrainerConfigDict:
        """Returns the final configuration."""
        restored = self._true_config
        del self._true_config
        return restored

    def validate_config(self, config: dict):
        """Assert final configurations are valid."""

    def get_policy_class(self) -> Type[Policy]:
        """Returns the Policy type to be set as the `_policy` attribute.

        Returns the `_policy` attribute by default. May be overriden to select a
        policy class depending on, e.g., the config dict.

        Called after :meth:`validate_config` and before :meth:`before_init`.
        """
        return self._policy_class

    def before_init(self):
        """Arbitrary setup before default initialization.

        Called after :meth:`get_policy_class` and before the creation of the
        worker set and execution plan.
        """

    @property
    def execution_plan(
        self,
    ) -> Callable[[WorkerSet, TrainerConfigDict], Iterable[ResultDict]]:
        """The execution plan function."""
        return default_execution_plan

    def after_init(self):
        """Arbitrary setup after default initialization.

        Called second-to-last in the :meth:`_init` procedure, after worker set
        and execution plan creation and before :meth:`optimize_policy_backend`.
        """

    def optimize_policy_backend(self):
        """Call `compile` on each policy if requested

        Called right after worker set creation. Requires the `compile` key under
        the `policy` subconfig.
        """
        if self.config["policy"].get("compile", False):
            self.workers.foreach_policy(lambda p, _: p.compile())

    @overrides(RLlibTrainer)
    def train(self) -> ResultDict:
        """Overrides super.train to synchronize global vars."""

        result = None
        for _ in range(1 + MAX_WORKER_FAILURE_RETRIES):
            try:
                result = Trainable.train(self)
            except RayError as err:
                if self.config["ignore_worker_failures"]:
                    logger.exception("Error in train call, attempting to recover")
                    self._try_recover()
                else:
                    logger.info(
                        "Worker crashed during call to train(). To attempt to "
                        "continue training without the failed worker, set "
                        "`'ignore_worker_failures': True`."
                    )
                    raise err
            except Exception as exc:
                time.sleep(0.5)  # allow logs messages to propagate
                raise exc
            else:
                break
        if result is None:
            raise RuntimeError("Failed to recover from worker crash")

        if hasattr(self, "workers") and isinstance(self.workers, WorkerSet):
            self._sync_filters_if_needed(self.workers)

        return result

    @overrides(RLlibTrainer)
    def step(self) -> dict:
        res = self.evaluate_pre_learning()
        res.update(next(self.train_exec_impl))
        res.update(self.evaluate_if_needed())
        return res

    def evaluate_pre_learning(self) -> dict:
        """Runs evaluation before any optimizations if requested.

        Returns:
            A dictionary with evaluation info
        """
        interval = self.config["evaluation_interval"]
        if self.iteration == 0 and interval and interval != 1:
            return self._evaluate()
        return {}

    def evaluate_if_needed(self) -> dict:
        """Runs evaluation episodes if configured to do so.

        Returns:
            A dictionary with evaluation info
        """
        iteration, interval = self.iteration + 1, self.config["evaluation_interval"]

        if interval == 1 or (iteration > 0 and interval and iteration % interval == 0):
            evaluation_metrics = self._evaluate()
            assert isinstance(
                evaluation_metrics, dict
            ), "_evaluate() needs to return a dict."
            return evaluation_metrics

        return {}

    @overrides(RLlibTrainer)
    def __getstate__(self) -> dict:
        state = super().__getstate__()
        state["train_exec_impl"] = self.train_exec_impl.shared_metrics.get().save()
        return state

    @overrides(RLlibTrainer)
    def __setstate__(self, state: dict):
        super().__setstate__(state)
        self.train_exec_impl.shared_metrics.get().restore(state["train_exec_impl"])

    @classmethod
    def default_resource_request(cls, config: PartialTrainerConfigDict) -> Resources:
        cnf = dict(cls.options.rllib_defaults, **config)
        cls._validate_config(cnf)
        num_workers = cnf["num_workers"] + cnf["evaluation_num_workers"]
        return Resources(
            cpu=cnf["num_cpus_for_driver"],
            gpu=cnf["num_gpus"],
            memory=cnf["memory"],
            object_store_memory=cnf["object_store_memory"],
            extra_cpu=cnf["num_cpus_per_worker"] * num_workers,
            extra_gpu=cnf["num_gpus_per_worker"] * num_workers,
            extra_memory=cnf["memory_per_worker"] * num_workers,
            extra_object_store_memory=cnf["object_store_memory_per_worker"]
            * num_workers,
        )
