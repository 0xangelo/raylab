from ray.tune.logger import TFLogger
from .config_json_logger import ConfigJsonLogger
from .flat_csv_logger import FlatCSVLogger

DEFAULT_LOGGERS = (ConfigJsonLogger, FlatCSVLogger, TFLogger)
