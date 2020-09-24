# pylint:disable=missing-module-docstring
from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import List
from typing import Tuple

from ray.rllib import SampleBatch

from raylab.options import option
from raylab.utils.timer import TimerStat
from raylab.utils.types import StatDict

from ..stats import learner_stats


def model_based_options(cls: type) -> type:
    """Decorator to add default model-based options used by MBPolicyMixin."""
    model_update_interval = option(
        "model_update_interval",
        default=1,
        help="""Number of calls to `learn_on_batch` between each model update run.

        Example:
            With a 'rollout_fragment_length' of 1 and 'model_update_interval' of 25,
            will collect 25 environment transitions between each model optimization
            loop.
        """,
    )
    for opt in [model_update_interval]:
        cls = opt(cls)

    return cls


class MBPolicyMixin(ABC):
    """Off-policy mixin with dynamics model learning."""

    timers: Dict[str, TimerStat]
    _learn_calls: int = 0
    _info: dict

    def build_timers(self):
        """Create timers for model and policy training."""
        self.timers = {"model": TimerStat(), "policy": TimerStat()}
        self._info = {}

    @learner_stats
    def learn_on_batch(self, samples: SampleBatch) -> dict:
        # pylint:disable=missing-function-docstring
        self.add_to_buffer(samples)
        self._learn_calls += 1

        warmup = self._learn_calls == 1
        if (self._learn_calls % self.config["model_update_interval"] == 0) or warmup:
            with self.timers["model"] as timer:
                _, model_info = self.train_dynamics_model(warmup=warmup)
                timer.push_units_processed(model_info["model_epochs"])
                self._info.update(model_info)

        with self.timers["policy"] as timer:
            times = self.config["improvement_steps"]
            policy_info = self.update_policy(times=times)
            timer.push_units_processed(times)
            self._info.update(policy_info)

        self._info.update(self.timer_stats())
        return self._info.copy()

    @abstractmethod
    def train_dynamics_model(
        self, warmup: bool = False
    ) -> Tuple[List[float], StatDict]:
        """Implements the model training step.

        Calls the policy to optimize the model on the environment replay buffer.

        Args:
            warmup: Whether the optimization is being done on data collected
                via :meth:`sample_until_learning_starts`.

        Returns:
            A tuple containing the list of evaluation losses for each model and
            a dictionary of training statistics
        """

    def update_policy(self, times: int) -> StatDict:
        """Improve the policy on previously collected environment data.

        Improves the policy on batches sampled from the replay buffer.

        Args:
            times: number of times to call :meth:`improve_policy`

        Returns:
            A dictionary of training statistics
        """
        for _ in range(times):
            batch = self.replay.sample(self.config["batch_size"])
            batch = self.lazy_tensor_dict(batch)
            info = self.improve_policy(batch)

        return info

    def timer_stats(self) -> dict:
        """Returns the timer statistics."""
        model_timer = self.timers["model"]
        policy_timer = self.timers["policy"]
        return dict(
            model_time_s=round(model_timer.mean, 3),
            policy_time_s=round(policy_timer.mean, 3),
            # Get mean number of model epochs per second spent updating the model
            model_update_throughput=round(model_timer.mean_throughput, 3),
            # Get mean number of policy updates per second spent updating the policy
            policy_update_throughput=round(policy_timer.mean_throughput, 3),
        )

    @staticmethod
    def add_options(policy_cls: type) -> type:
        """Decorator to add default model-based options used by MBPolicyMixin."""
        return model_based_options(policy_cls)
