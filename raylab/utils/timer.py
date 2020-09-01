# pylint:disable=missing-module-docstring
from ray.util.timer import _Timer


class TimerStat(_Timer):
    """Ray timer that returns itself upon entering a context."""

    def __enter__(self) -> "TimerStat":
        super().__enter__()
        return self
