"""Custom loggers to use with Tune."""
from ray.tune.logger import CSVLogger

from .progress_json_logger import ProgressJsonLogger

DEFAULT_LOGGERS = (ProgressJsonLogger, CSVLogger)

try:
    from .torch_tensorboard_logger import TorchTBLogger

    DEFAULT_LOGGERS = DEFAULT_LOGGERS + (TorchTBLogger,)
except ImportError:
    pass
