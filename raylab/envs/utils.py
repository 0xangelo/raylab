"""Env creation utilities."""
import functools
import importlib
import inspect
from typing import Callable
from typing import Mapping

import gym
from gym.wrappers import TimeLimit
from ray.tune.registry import _global_registry
from ray.tune.registry import ENV_CREATOR

from .wrappers import AddRelativeTimestep
from .wrappers import SinglePrecision


def has_env_creator(env_id: str) -> bool:
    "Whether and environment with the given id is in the global registry."
    return _global_registry.contains(ENV_CREATOR, env_id)


def get_env_creator(env_id: str) -> Callable[[dict], gym.Env]:
    """Return the environment creator funtion for the given environment id."""
    if not _global_registry.contains(ENV_CREATOR, env_id):
        raise ValueError(f"Environment id {env_id} not registered in Tune")
    return _global_registry.get(ENV_CREATOR, env_id)


def wrap_if_needed(env_creator):
    """Wraps an env creator function to handle time limit configurations."""

    @functools.wraps(env_creator)
    def wrapped(config):
        tmp = config.copy()
        time_limit = [tmp.pop(k, None) for k in ("time_aware", "max_episode_steps")]
        single_precision = tmp.pop("single_precision", False)
        env = env_creator(tmp)
        env = wrap_time_limit(env, *time_limit)
        if single_precision:
            env = SinglePrecision(env)
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
            # pylint:disable=protected-access
            env_._max_episode_steps = max_episode_steps
            # pylint: enable=protected-access
        else:
            env = TimeLimit(env, max_episode_steps=max_episode_steps)

    if time_aware:
        env = AddRelativeTimestep(env)

    return env


def get_env_parameters(env_id: str) -> Mapping[str, inspect.Parameter]:
    """Return an unwrapped environment's constructor parameters.

    Args:
        env_id: The environment name registered in Gym

    Returns:
        A mapping from parameter names to Parameter objects
    """
    env_spec = gym.spec(env_id)
    mod_str, cls_str = env_spec.entry_point.split(":")
    mod = importlib.import_module(mod_str)
    env_cls = getattr(mod, cls_str)

    return inspect.signature(env_cls).parameters
