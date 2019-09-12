# pylint: disable=missing-docstring
from ray.tune.logger import JsonLogger as _JsonLogger
from ray.rllib.utils.annotations import override


class ConfigJsonLogger(_JsonLogger):
    """Custom JsonLogger which only saves the trial configuration."""

    @override(_JsonLogger)
    def _init(self):
        self.update_config(self.config)

    @override(_JsonLogger)
    def on_result(self, result):
        pass
