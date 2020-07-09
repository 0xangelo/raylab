# pylint:disable=missing-module-docstring
from ray.rllib.utils import override
from ray.tune.logger import JsonLogger as _JsonLogger


class ProgressJsonLogger(_JsonLogger):
    """Custom JsonLogger which does not save the trial configuration every time."""

    @override(_JsonLogger)
    def on_result(self, result):
        tmp = result.copy()
        if "config" in tmp:
            del tmp["config"]
        super().on_result(tmp)
