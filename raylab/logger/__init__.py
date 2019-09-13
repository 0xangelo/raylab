"""Custom loggers to use with Tune."""
from ray.tune.logger import CSVLogger
from raylab.logger.config_json_logger import ConfigJsonLogger
from raylab.logger.torch_tensorboard_logger import TorchTBLogger

DEFAULT_LOGGERS = (ConfigJsonLogger, CSVLogger, TorchTBLogger)
