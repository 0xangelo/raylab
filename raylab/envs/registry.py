"""Registry of custom Gym environments."""
from .utils import env_maker, add_time_limit


def _cartpole_swingup_maker(_):
    from raylab.envs.cartpole_swingup import CartPoleSwingUpEnv

    return CartPoleSwingUpEnv()


def _cartpole_stateless_maker(_):
    from gym.envs.classic_control.cartpole import CartPoleEnv
    from raylab.envs.cartpole_stateless import CartPoleStatelessWrapper

    return CartPoleStatelessWrapper(CartPoleEnv())


def _time_limited_env_maker(config):
    from raylab.envs.time_aware_env import AddRelativeTimestep
    from raylab.envs.peb_env import IgnoreTimeoutTerminations

    env = env_maker(config["env_id"])(**config)
    env = add_time_limit(env, config["max_episode_steps"])
    if config.get("time_aware", False):
        return AddRelativeTimestep(env)
    return IgnoreTimeoutTerminations(env)


def _navigation_maker(config):
    from raylab.envs.navigation import NavigationEnv

    return NavigationEnv(config)


ENVS = {
    "CartPoleSwingUp": _cartpole_swingup_maker,
    "CartPoleStateless": _cartpole_stateless_maker,
    "TimeLimitedEnv": _time_limited_env_maker,
    "Navigation": _navigation_maker,
}
