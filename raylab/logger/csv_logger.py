"""Custom CSV logger to use with Tune."""
import csv

from ray.tune.util import flatten_dict
from ray.tune.logger import CSVLogger as _CSVLogger
from ray.tune.result import TRAINING_ITERATION, TIME_TOTAL_S
from ray.rllib.utils.annotations import override


class CSVLogger(_CSVLogger):
    """CSV logger that removes unuseful keys before logging."""

    @override(_CSVLogger)
    def on_result(self, result):
        tmp = result.copy()
        elim = ["config", "pid", "timestamp", TIME_TOTAL_S, TRAINING_ITERATION]
        for k in elim:
            if k in tmp:
                del tmp[k]  # not useful to tf log these
        result = flatten_dict(tmp, delimiter="/")
        if self._csv_out is None:
            # pylint: disable=attribute-defined-outside-init
            self._csv_out = csv.DictWriter(self._file, result.keys())
            # pylint: enable=attribute-defined-outside-init
            if not self._continuing:
                self._csv_out.writeheader()
        self._csv_out.writerow(
            {k: v for k, v in result.items() if k in self._csv_out.fieldnames}
        )
        self._file.flush()
