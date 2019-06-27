from ray.tune.util import flatten_dict
from ray.tune.logger import CSVLogger


class FlatCSVLogger(CSVLogger):
    def on_result(self, result):
        tmp = result.copy()
        if "config" in tmp:
            del tmp["config"]
        super(FlatCSVLogger, self).on_result(flatten_dict(tmp))
