"""Registry of custom Gym environments."""
from .utils import env_maker, add_time_limit


def _cartpole_swingup_maker(config):
    from raylab.envs.cartpole_swingup import CartPoleSwingUpEnv

    env = CartPoleSwingUpEnv()
    if config.get("max_episode_steps", False):
        env = add_time_limit(env, config["max_episode_steps"])
        if config.get("time_aware", False):
            from raylab.envs.time_aware_env import AddRelativeTimestep

            env = AddRelativeTimestep(env)
    return env


def _cartpole_stateless_maker(_):
    from gym.envs.classic_control.cartpole import CartPoleEnv
    from raylab.envs.cartpole_stateless import CartPoleStatelessWrapper

    return CartPoleStatelessWrapper(CartPoleEnv())


def _time_aware_env_maker(config):
    from raylab.envs.time_aware_env import AddRelativeTimestep

    env = env_maker(config["env_id"])(**config)
    env = add_time_limit(env, config["max_episode_steps"])
    return AddRelativeTimestep(env)


def _navigation_maker(config):
    from raylab.envs.navigation import NavigationEnv

    return NavigationEnv(config)


ENVS = {
    "CartPoleSwingUp": _cartpole_swingup_maker,
    "CartPoleStateless": _cartpole_stateless_maker,
    "TimeAwareEnv": _time_aware_env_maker,
    "Navigation": _navigation_maker,
}
