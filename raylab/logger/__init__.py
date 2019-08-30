from ray.tune.logger import TFLogger
from raylab.logger.config_json_logger import ConfigJsonLogger
from raylab.logger.flat_csv_logger import FlatCSVLogger

DEFAULT_LOGGERS = (ConfigJsonLogger, FlatCSVLogger, TFLogger)
