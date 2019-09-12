"""Custom loggers to use with Tune."""
from ray.tune.logger import TFLogger, CSVLogger
from raylab.logger.config_json_logger import ConfigJsonLogger

DEFAULT_LOGGERS = (ConfigJsonLogger, CSVLogger, TFLogger)
