# pylint:disable=missing-module-docstring
import contextlib
import functools
import os


def context_to_devnull(func, *, context_fn):
    """Decorator for applying a file-dependent context manager with devnull."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with open(os.devnull, "w") as devnull:
            with context_fn(devnull):
                return func(*args, **kwargs)

    return wrapper


supress_stdout = functools.partial(
    context_to_devnull, context_fn=contextlib.redirect_stdout
)
supress_stderr = functools.partial(
    context_to_devnull, context_fn=contextlib.redirect_stderr
)
