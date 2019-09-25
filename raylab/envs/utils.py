"""Env creation utilities."""
import gym
from gym.wrappers import TimeLimit
from ray.tune.registry import _global_registry, ENV_CREATOR


def env_maker(env_id):
    """Return an environment maker function.

    By default, tries to fetch makers in Tune's global registry. If not present,
    uses `gym.make`.

    Arguments:
        env_id (str): name of the environment

    Returns:
        A callable with a single config argument
    """
    if _global_registry.contains(ENV_CREATOR, env_id):
        return _global_registry.get(ENV_CREATOR, env_id)
    return lambda _: gym.make(env_id)


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
