# pylint:disable=missing-docstring

import functools


def initialize_raylab(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        import raylab

        raylab.register_all_agents()
        raylab.register_all_environments()

        return func(*args, **kwargs)

    return wrapped
