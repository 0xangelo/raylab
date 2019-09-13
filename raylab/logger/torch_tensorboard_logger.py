# pylint: disable=missing-docstring
from torch.utils.tensorboard import SummaryWriter
from ray.tune.util import flatten_dict
from ray.tune.logger import TFLogger
from ray.tune.result import TRAINING_ITERATION, TIME_TOTAL_S, TIMESTEPS_TOTAL
from ray.rllib.utils.annotations import override


class TorchTBLogger(TFLogger):
    """Tensorboard logger with no tensorflow dependency."""

    @override(TFLogger)
    def _init(self):
        self._file_writer = SummaryWriter(self.logdir)

    @override(TFLogger)
    def on_result(self, result):
        tmp = result.copy()
        for k in ["config", "pid", "timestamp", TIME_TOTAL_S, TRAINING_ITERATION]:
            if k in tmp:
                del tmp[k]  # not useful to tf log these

        values = flatten_dict(tmp, delimiter="/")
        values.update({TRAINING_ITERATION: result[TRAINING_ITERATION]})
        step = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]

        self._file_writer.add_scalars("ray/tune/", values, step)
        self._file_writer.flush()
