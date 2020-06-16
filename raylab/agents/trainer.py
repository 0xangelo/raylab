"""Primitives for all Trainers."""
import warnings
from abc import ABCMeta
from dataclasses import dataclass
from dataclasses import field
from typing import List
from typing import Optional
from typing import Union

from ray.rllib.agents import with_common_config as with_rllib_config
from ray.rllib.agents.trainer import Trainer as _Trainer
from ray.rllib.agents.trainer import with_base_config
from ray.rllib.evaluation.metrics import collect_episodes
from ray.rllib.evaluation.metrics import summarize_episodes
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.optimizers import PolicyOptimizer
from ray.rllib.utils import override

_Trainer._allow_unknown_subkeys += ["module", "torch_optimizer"]
_Trainer._override_all_subkeys_if_type_changes += ["module"]


BASE_CONFIG = with_rllib_config(
    {
        # === Policy ===
        # Whether to optimize the policy's backend
        "compile_policy": False
    }
)


def with_common_config(extra_config: dict) -> dict:
    """Returns the given config dict merged with common agent confs."""
    return with_base_config(BASE_CONFIG, extra_config)


@dataclass
class StatsTracker:
    """Emulates the metric logging behavior of RLlib's PolicyOptimizer.

    Attributes:
        workers: The set of rollout workers to track.
        num_steps_trained: Number of timesteps trained on so far.
        num_steps_sampled: Number of timesteps sampled so far.
    """

    workers: Optional[WorkerSet] = None
    num_steps_trained: int = field(default=0, init=False)
    num_steps_sampled: int = field(default=0, init=False)

    _episode_history: list = field(default_factory=list)
    _to_be_collected: list = field(default_factory=list)

    def __post_init__(self):
        if not self.workers:
            warnings.warn(
                "No worker set provided to stats tracker. Episodes summary "
                "will be unavailable."
            )

    def save(self) -> List[int]:
        """Returns a serializable object representing the optimizer state."""

        return [self.num_steps_trained, self.num_steps_sampled]

    def restore(self, data: List[int]):
        """Restores optimizer state from the given data object."""

        self.num_steps_trained = data[0]
        self.num_steps_sampled = data[1]

    def collect_metrics(
        self,
        timeout_seconds: int,
        min_history: int = 100,
        selected_workers: Optional[list] = None,
    ) -> dict:
        """Returns worker stats.

        Args:
            timeout_seconds: Max wait time for a worker before
                dropping its results. This usually indicates a hung worker.
            min_history: Min history length to smooth results over.
            selected_workers: Override the list of remote workers
                to collect metrics from.

        Returns:
            res: A training result dict from worker metrics with
                `info` replaced with stats from self.
        """
        res = {}
        if self.workers:
            episodes, self._to_be_collected = collect_episodes(
                self.workers.local_worker(),
                selected_workers or self.workers.remote_workers(),
                self._to_be_collected,
                timeout_seconds=timeout_seconds,
            )
            orig_episodes = list(episodes)
            missing = min_history - len(episodes)
            if missing > 0:
                episodes.extend(self._episode_history[-missing:])
                assert len(episodes) <= min_history
            self._episode_history.extend(orig_episodes)
            self._episode_history = self._episode_history[-min_history:]
            res = summarize_episodes(episodes, orig_episodes)
        return res

    @staticmethod
    def stop():
        """Placeholder to emulate PolicyOptimizer.save."""


class Trainer(_Trainer, metaclass=ABCMeta):
    """Base Trainer for all agents.

    Either a StatsTracker, PolicyOptimizer, or WorkerSet must be set (as
    `tracker`, `optimizer`, or `workers` attributes respectively) to collect
    episode statistics.

    If a PolicyOptimizer is set, adds a `tracker` attribute pointing to it
    so that logging code is standardized.
    """

    evaluation_metrics: Optional[dict]
    optimizer: Optional[PolicyOptimizer]
    workers: Optional[WorkerSet]
    tracker: Union[StatsTracker, PolicyOptimizer]

    def _setup(self, *args, **kwargs):
        super()._setup(*args, **kwargs)
        if hasattr(self, "tracker"):
            pass
        elif hasattr(self, "optimizer"):
            self.tracker = self.optimizer
        elif hasattr(self, "workers"):
            self.tracker = StatsTracker(self.workers)
        else:
            self.tracker = StatsTracker()

        # Needed for train() to synchronize global_vars
        if not hasattr(self, "optimizer"):
            self.optimizer = self.tracker

        if self.config["compile_policy"]:
            if hasattr(self, "workers"):
                workers = self.workers
            elif hasattr(self, "tracker") and hasattr(self.tracker, "workers"):
                workers = self.tracker.workers
            else:
                raise RuntimeError(
                    f"{type(self).__name__} has no worker set. "
                    "Cannot access policies for compilation."
                )
            workers.foreach_policy(lambda p, _: p.compile())

    @override(_Trainer)
    def train(self):
        # Evaluate first, before any optimization is done
        if self._iteration == 0 and self.config["evaluation_interval"]:
            self.evaluation_metrics = self._evaluate()

        result = super().train()

        # Update global_vars after training so that they're saved if checkpointing
        self.global_vars["timestep"] = self.tracker.num_steps_sampled
        return result

    @override(_Trainer)
    def collect_metrics(self, selected_workers=None):
        return self.tracker.collect_metrics(
            self.config["collect_metrics_timeout"],
            min_history=self.config["metrics_smoothing_episodes"],
            selected_workers=selected_workers,
        )

    @override(_Trainer)
    def __getstate__(self):
        state = super().__getstate__()
        state["global_vars"] = self.global_vars

        if not hasattr(self, "optimizer"):
            state["tracker"] = self.tracker.save()
        return state

    @override(_Trainer)
    def __setstate__(self, state):
        self.global_vars = state["global_vars"]

        if self.tracker.workers:
            self.tracker.workers.foreach_worker(
                lambda w: w.set_global_vars(self.global_vars)
            )

        if "optimizer" not in state:
            self.tracker.restore(state["tracker"])

        super().__setstate__(state)

    def _iteration_done(self, init_timesteps):
        return self.tracker.num_steps_sampled - init_timesteps >= max(
            self.config["timesteps_per_iteration"], 1
        )

    def _log_metrics(self, learner_stats, init_timesteps):
        res = self.collect_metrics()
        res.update(
            timesteps_this_iter=self.tracker.num_steps_sampled - init_timesteps,
            info=dict(learner=learner_stats, **res.get("info", {})),
        )
        if self._iteration == 0 and self.config["evaluation_interval"]:
            res.update(self.evaluation_metrics)
        return res
