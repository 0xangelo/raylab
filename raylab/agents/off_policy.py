# pylint:disable=missing-module-docstring
from typing import Callable
from typing import Iterable

from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.execution.rollout_ops import ParallelRollouts
from ray.rllib.execution.train_ops import TrainOneStep
from ray.rllib.utils.typing import ResultDict
from ray.rllib.utils.typing import TrainerConfigDict

from raylab.execution import LearningStarts
from raylab.options import option


def off_policy_execution_plan(workers: WorkerSet, config: TrainerConfigDict):
    """RLlib's default execution plan with an added warmup phase."""
    # Collects experiences in parallel from multiple RolloutWorker actors.
    rollouts = ParallelRollouts(workers, mode="bulk_sync")
    # On the first iteration, combine experience batches until we hit `learning_starts`
    # in size.
    rollouts = rollouts.combine(
        LearningStarts(learning_starts=config["learning_starts"])
    )
    # Then, train the policy on those experiences and update the workers.
    train_op = rollouts.for_each(TrainOneStep(workers))

    # Add on the standard episode reward, etc. metrics reporting. This returns
    # a LocalIterator[metrics_dict] representing metrics for each train step.
    return StandardMetricsReporting(train_op, workers, config)


class OffPolicyMixin:
    """Mixin for off-policy agents."""

    # pylint:disable=missing-function-docstring
    def validate_config(self, config: dict):
        super().validate_config(config)
        assert config["num_workers"] == 0, "No point in using additional workers."
        assert (
            config["rollout_fragment_length"] >= 1
        ), "At least one sample must be collected."

    @property
    def execution_plan(
        self,
    ) -> Callable[[WorkerSet, TrainerConfigDict], Iterable[ResultDict]]:
        return off_policy_execution_plan

    @staticmethod
    def add_options(trainer_cls: type) -> type:
        cls = trainer_cls
        for opt in [
            option(
                "learning_starts",
                default=0,
                help="Hold this number of timesteps before first training operation.",
            ),
            option("rollout_fragment_length", default=1, override=True),
            option("num_workers", default=0, override=True),
            option("evaluation_config/explore", False, override=True),
        ]:
            cls = opt(cls)
        return cls
