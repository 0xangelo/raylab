# pylint:disable=missing-module-docstring
import contextlib
import functools
import logging
import os
import warnings


@contextlib.contextmanager
def lightning_warnings_only():
    logger = logging.getLogger("lightning")
    level = logger.getEffectiveLevel()
    try:
        logger.setLevel(logging.WARNING)
        yield
    finally:
        logger.setLevel(level)


@contextlib.contextmanager
def suppress_dataloader_warnings():
    """Ignore PyTorch Lightning warnings regarding num of dataloader workers."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*Consider increasing the value of the `num_workers`.*",
            module="pytorch_lightning.utilities.distributed",
        )
        warnings.filterwarnings(
            "ignore",
            message="One of given dataloaders is None.*",
            module="pytorch_lightning.utilities.distributed",
        )
        yield


def context_to_devnull(func, *, context_fn):
    """Decorator for applying a file-dependent context manager with devnull."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with open(os.devnull, "w") as devnull:
            with context_fn(devnull):
                return func(*args, **kwargs)

    return wrapper


suppress_stdout = functools.partial(
    context_to_devnull, context_fn=contextlib.redirect_stdout
)
suppress_stderr = functools.partial(
    context_to_devnull, context_fn=contextlib.redirect_stderr
)
