# pylint:disable=missing-module-docstring
import logging
import time
from abc import ABCMeta
from abc import abstractmethod
from typing import Callable
from typing import Iterable
from typing import Optional
from typing import Type

from ray.exceptions import RayError
from ray.rllib import Policy
from ray.rllib.agents import Trainer as RLlibTrainer
from ray.rllib.agents.trainer import MAX_WORKER_FAILURE_RETRIES
from ray.rllib.env.env_context import EnvContext
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.execution.rollout_ops import ConcatBatches
from ray.rllib.execution.rollout_ops import ParallelRollouts
from ray.rllib.execution.train_ops import TrainOneStep
from ray.rllib.utils import override as overrides
from ray.rllib.utils.types import EnvType
from ray.rllib.utils.types import PartialTrainerConfigDict
from ray.rllib.utils.types import ResultDict
from ray.rllib.utils.types import TrainerConfigDict
from ray.tune.trainable import Trainable

from raylab.execution import LearningStarts
from raylab.options import configure
from raylab.options import option
from raylab.options import TrainerOptions
from raylab.utils.wandb import WandBLogger

logger = logging.getLogger(__name__)


# ==============================================================================
# Default Execution Plan
# ==============================================================================
def default_execution_plan(workers: WorkerSet, config: TrainerConfigDict):
    """RLlib's default execution plan with an added warmup phase."""
    # Collects experiences in parallel from multiple RolloutWorker actors.
    rollouts = ParallelRollouts(workers, mode="bulk_sync")

    # On the first iteration, hold until we have at least `learning_starts` timesteps.
    # Combine experiences batches until we hit `train_batch_size` in size.
    # Then, train the policy on those experiences and update the workers.
    train_op = (
        rollouts.combine(LearningStarts(learning_starts=config["learning_starts"]))
        .combine(ConcatBatches(min_batch_size=config["train_batch_size"]))
        .for_each(TrainOneStep(workers))
    )

    # Add on the standard episode reward, etc. metrics reporting. This returns
    # a LocalIterator[metrics_dict] representing metrics for each train step.
    return StandardMetricsReporting(train_op, workers, config)


# ==============================================================================
# Streamlined Trainer
# ==============================================================================
@configure
@option(
    "policy/",
    allow_unknown_subkeys=True,
    help="""Sub-configurations for the policy class.""",
)
@option(
    "wandb/",
    allow_unknown_subkeys=True,
    help="""Configs for integration with Weights & Biases.

    Accepts arbitrary keyword arguments to pass to `wandb.init`.
    The defaults for `wandb.init` are:
    * name: `_name` property of the trainer.
    * config: full `config` attribute of the trainer
    * config_exclude_keys: `wandb` and `callbacks` configs
    * reinit: True

    Don't forget to:
      * install `wandb` via pip
      * login to W&B with the appropriate API key for your
        team/project.
      * set the `wandb/project` name in the config dict

    Check out the Quickstart for more information:
    `https://docs.wandb.com/quickstart`
    """,
)
@option(
    "learning_starts",
    default=0,
    help="""Hold this number of timesteps before first training operation.""",
)
class SimpleTrainer(RLlibTrainer, metaclass=ABCMeta):
    # pylint:disable=missing-docstring
    config: TrainerConfigDict
    raw_user_config: PartialTrainerConfigDict
    env_creator: Callable[[EnvContext], EnvType]
    global_vars: dict
    workers: WorkerSet
    evaluation_workers: Optional[WorkerSet]
    train_exec_impl: Iterable[ResultDict]
    wandb: WandBLogger
    options: TrainerOptions = TrainerOptions()
    _name: str
    _policy: Type[Policy]
    _env_id: str
    _true_config: TrainerConfigDict

    def setup(self, config: PartialTrainerConfigDict):
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
        self._policy = cls = self.get_policy_class(config)

        self.before_init()

        # Creating all workers (excluding evaluation workers).
        num_workers = config["num_workers"]
        self.workers = self._make_workers(env_creator, cls, config, num_workers)
        self.optimize_policy_backend()
        self.train_exec_impl = self.execution_plan(self.workers, config)
        self.wandb = WandBLogger(config, self._name)

        self.after_init()

    def restore_reserved(self) -> TrainerConfigDict:
        restored = self._true_config
        del self._true_config
        return restored

    @staticmethod
    def validate_config(config: dict):
        pass

    @abstractmethod
    def get_policy_class(self, config: dict) -> Type[Policy]:
        """Returns the Policy type to be set as the `_policy` attribute.

        Called after :meth:`validate_config` and before :meth:`before_init`.
        """

    def before_init(self):
        """Arbitrary setup before default initialization.

        Called after :meth:`get_policy_class` and before the creation of the
        worker set, execution plan, and wandb logger.
        """

    @property
    def execution_plan(
        self,
    ) -> Callable[[WorkerSet, TrainerConfigDict], Iterable[ResultDict]]:
        """The execution plan function."""
        return default_execution_plan

    def after_init(self):
        """Arbitrary setup after default initialization.

        Called last in the :meth:`_init` procedure, after the worker set,
        execution plan, and wandb logger have been created.
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

        print(f"OLD EVAL ITER: {self.iteration}")
        return result

    @overrides(RLlibTrainer)
    def step(self) -> dict:
        res = next(self.train_exec_impl)
        res = self.evaluate_if_needed(res)
        return res

    def evaluate_if_needed(self, result: dict) -> dict:
        iteration, interval = self.iteration + 1, self.config["evaluation_interval"]
        print(f"NEW EVAL ITER: {iteration}")
        if interval == 1 or (iteration > 0 and interval and iteration % interval == 0):
            evaluation_metrics = self._evaluate()
            assert isinstance(
                evaluation_metrics, dict
            ), "_evaluate() needs to return a dict."
            return {**result, **evaluation_metrics}

        return result

    @overrides(RLlibTrainer)
    def log_result(self, result: ResultDict):
        super().log_result(result)
        if self.wandb.enabled:
            self.wandb.log_result(result)

    @overrides(RLlibTrainer)
    def __getstate__(self) -> dict:
        state = super().__getstate__()
        state["train_exec_impl"] = self.train_exec_impl.shared_metrics.get().save()
        return state

    @overrides(RLlibTrainer)
    def __setstate__(self, state: dict):
        super().__setstate__(state)
        self.train_exec_impl.shared_metrics.get().restore(state["train_exec_impl"])

    @overrides(RLlibTrainer)
    def cleanup(self):
        super().cleanup()
        if self.wandb.enabled:
            self.wandb.stop()
