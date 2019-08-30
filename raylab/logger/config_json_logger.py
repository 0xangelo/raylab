import os
import json

from ray.tune.logger import Logger, _SafeFallbackEncoder
import ray.cloudpickle as cloudpickle


class ConfigJsonLogger(Logger):
    def _init(self):
        self.update_config(self.config)

    def on_result(self, result):
        pass

    def update_config(self, config):
        self.config = config
        config_out = os.path.join(self.logdir, "params.json")
        with open(config_out, "w") as file:
            json.dump(self.config, file, cls=_SafeFallbackEncoder)
        config_pkl = os.path.join(self.logdir, "params.pkl")
        with open(config_pkl, "wb") as file:
            cloudpickle.dump(self.config, file)
