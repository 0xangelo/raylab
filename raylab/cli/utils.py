# pylint:disable=missing-module-docstring
import functools


def initialize_raylab(func):
    """Wrap cli to register raylab's algorithms and environments."""

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        import raylab

        raylab.register_all_agents()
        raylab.register_all_environments()

        return func(*args, **kwargs)

    return wrapped
