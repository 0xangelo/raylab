# pylint:disable=missing-module-docstring
import logging
from typing import List

from ray.rllib import SampleBatch
from ray.rllib.execution.common import (
    STEPS_SAMPLED_COUNTER,
    _check_sample_batch_type,
    _get_shared_metrics,
)
from ray.rllib.utils.typing import SampleBatchType

logger = logging.getLogger(__name__)


class LearningStarts:
    """Callable used to merge initial batches until a desired timestep count is met

    This should be used with the .combine() operator.

    Examples:
        >>> rollouts = ParallelRollouts(...)
        >>> rollouts = rollouts.combine(LearningStarts(learning_starts=10000))
        >>> print(next(rollouts).count)
        10000
    """

    # pylint:disable=too-few-public-methods
    def __init__(self, learning_starts: int):
        self.learning_starts = learning_starts
        self.buffer = []
        self.count = 0
        self.done = False

    def __call__(self, batch: SampleBatchType) -> List[SampleBatchType]:
        _check_sample_batch_type(batch)
        if self.done:
            # Warmup phase done, simply return batch
            return [batch]

        metrics = _get_shared_metrics()
        timesteps_total = metrics.counters[STEPS_SAMPLED_COUNTER]
        self.buffer.append(batch)
        self.count += batch.count
        assert self.count == timesteps_total

        if timesteps_total < self.learning_starts:
            # Return emtpy if still in warmup
            return []

        # Warmup just done
        if self.count > self.learning_starts * 2:
            logger.info(  # pylint:disable=logging-fstring-interpolation
                "Collected more training samples than expected "
                f"(actual={self.count}, expected={self.learning_starts}). "
                "This may be because you have many workers or "
                "long episodes in 'complete_episodes' batch mode."
            )
        out = SampleBatch.concat_samples(self.buffer)
        self.buffer = []
        self.count = 0
        self.done = True
        return [out]
