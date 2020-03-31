"""Custom loggers to use with Tune."""
from raylab.logger.config_json_logger import ConfigJsonLogger
from raylab.logger.csv_logger import CSVLogger

DEFAULT_LOGGERS = (ConfigJsonLogger, CSVLogger)

try:
    from raylab.logger.torch_tensorboard_logger import TorchTBLogger

    DEFAULT_LOGGERS = DEFAULT_LOGGERS + (TorchTBLogger,)
except ImportError:
    pass
