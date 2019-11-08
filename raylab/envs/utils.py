"""Env creation utilities."""
import functools

from gym.wrappers import TimeLimit

from .time_aware_env import AddRelativeTimestep
from .gaussian_random_walks import GaussianRandomWalks


def wrap_if_needed(env_creator):
    """Wraps an env creator function to handle time limit configurations."""

    @functools.wraps(env_creator)
    def wrapped(config):
        env = env_creator(config)
        env = wrap_time_limit(
            env, config.get("time_aware"), config.get("max_episode_steps")
        )
        env = wrap_gaussian_random_walks(env, config.get("random_walks"))
        return env

    return wrapped


def wrap_time_limit(env, time_aware, max_episode_steps):
    """Add or update an environment's time limit

    Arguments:
        env (gym.Env): a gym environment instance
        time_aware (bool): whether to append the relative timestep to the observation.
        max_episode_steps (int): the maximum number of timesteps in a single episode

    Returns:
        A wrapped environment with the desired time limit
    """
    assert not time_aware or max_episode_steps, "Time-aware envs must specify a horizon"

    if max_episode_steps:
        env_, has_timelimit = env, False
        while hasattr(env_, "env"):
            if isinstance(env_, TimeLimit):
                has_timelimit = True
                break
            env_ = env_.env

        if has_timelimit:
            # pylint: disable=protected-access
            env_._max_episode_steps = max_episode_steps
            # pylint: enable=protected-access
        else:
            env = TimeLimit(env, max_episode_steps=max_episode_steps)

    if time_aware:
        env = AddRelativeTimestep(env)

    return env


def wrap_gaussian_random_walks(env, walks_kwargs):
    """Add gaussian random walk variables to the observations, if specified.

    Arguments:
        env (gym.Env): a gym environment instance
        walks_kwargs (dict): arguments to pass to GaussianRandomWalks wrapper.

    Returns:
        A wrapped environment with the desired number of random walks
    """
    if walks_kwargs:
        env = GaussianRandomWalks(env, **walks_kwargs)
    return env
