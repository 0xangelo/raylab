"""Env creation utilities."""
import functools

from gym.wrappers import TimeLimit

from .time_aware_env import AddRelativeTimestep


def wrap_if_needed(env_creator):
    """Wraps an env creator function to handle time limit configurations."""

    @functools.wraps(env_creator)
    def wrapped(config):
        time_aware = config.pop("time_aware", False)
        max_episode_steps = config.pop("max_episode_steps", None)
        assert not time_aware or max_episode_steps

        env = env_creator(config)
        if max_episode_steps:
            env = add_time_limit(env, max_episode_steps)
        if time_aware:
            env = AddRelativeTimestep(env)
        return env

    return wrapped


def add_time_limit(env, max_episode_steps):
    """Add or update an environment's time limit

    Arguments:
        env (gym.Env): a gym environment instance
        max_episode_steps (int): the maximum number of timesteps in a single episode

    Returns:
        A wrapped environment with the desired time limit
    """
    _env, has_timelimit = env, False
    while hasattr(_env, "env"):
        if isinstance(_env, TimeLimit):
            has_timelimit = True
            break
        _env = _env.env

    if has_timelimit:
        # pylint: disable=protected-access
        _env._max_episode_steps = max_episode_steps
        # pylint: enable=protected-access
    else:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env
