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
    ignore = functools.partial(
        warnings.filterwarnings,
        "ignore",
        module="pytorch_lightning.trainer.data_loading",
    )
    with warnings.catch_warnings():
        ignore(message="The dataloader, .+, does not have many workers")
        ignore(message="One of given dataloaders is None")
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
