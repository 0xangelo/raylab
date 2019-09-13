# pylint: disable=missing-docstring
from torch.utils.tensorboard import SummaryWriter
from ray.tune.util import flatten_dict
from ray.tune.logger import Logger, VALID_SUMMARY_TYPES
from ray.tune.result import TRAINING_ITERATION, TIME_TOTAL_S, TIMESTEPS_TOTAL
from ray.rllib.utils.annotations import override


class TorchTBLogger(Logger):
    """Tensorboard logger with no tensorflow dependency."""

    @override(Logger)
    def _init(self):
        self._file_writer = SummaryWriter(self.logdir)

    @override(Logger)
    def on_result(self, result):
        tmp = result.copy()
        for k in ["config", "pid", "timestamp", TIME_TOTAL_S, TRAINING_ITERATION]:
            if k in tmp:
                del tmp[k]

        step = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]
        tmp.update({TRAINING_ITERATION: result[TRAINING_ITERATION]})
        for key, val in flatten_dict(tmp).items():
            if isinstance(val, tuple(VALID_SUMMARY_TYPES)):
                self._file_writer.add_scalar("/".join(["ray", "tune", key]), val, step)

        self._file_writer.flush()

    @override(Logger)
    def flush(self):
        self._file_writer.flush()

    @override(Logger)
    def close(self):
        self._file_writer.close()
