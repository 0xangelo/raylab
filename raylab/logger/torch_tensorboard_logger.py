# pylint:disable=missing-docstring
from ray.rllib.utils import override
from ray.tune.logger import Logger
from ray.tune.logger import VALID_SUMMARY_TYPES
from ray.tune.result import TIME_TOTAL_S
from ray.tune.result import TIMESTEPS_TOTAL
from ray.tune.result import TRAINING_ITERATION
from ray.tune.utils.util import flatten_dict
from torch.utils.tensorboard import SummaryWriter


class TorchTBLogger(Logger):
    """Tensorboard logger with no tensorflow dependency."""

    @override(Logger)
    def _init(self):
        self._file_writer = SummaryWriter(self.logdir)

    @override(Logger)
    def on_result(self, result):
        tmp = result.copy()
        elim = ["config", "pid", "timestamp", "done", TIME_TOTAL_S, TRAINING_ITERATION]
        for k in elim:
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
