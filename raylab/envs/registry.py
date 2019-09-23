"""Registry of custom Gym environments."""


def _cartpole_swingup_maker(_):
    import math
    import torch
    from gym_cartpole_swingup.envs import CartPoleSwingUpEnv

    def reward_fn(state, action, next_state):  # pylint: disable=unused-argument
        reward_theta = (torch.cos(next_state[..., 2]) + 1.0) / 2.0
        reward_x = torch.cos((next_state[..., 0] / 2.4) * (math.pi / 2.0))
        return reward_theta * reward_x

    env = CartPoleSwingUpEnv()
    setattr(env, "reward_fn", reward_fn)
    return env


def _cartpole_stateless_maker(_):
    from gym.envs.classic_control.cartpole import CartPoleEnv
    from raylab.envs.cartpole_stateless import CartPoleStatelessWrapper

    return CartPoleStatelessWrapper(CartPoleEnv())


def _time_aware_env_maker(config):
    import gym
    from gym.wrappers import TimeLimit
    from ray.tune.registry import _global_registry, ENV_CREATOR
    from raylab.envs.time_aware_env import AddRelativeTimestep

    if _global_registry.contains(ENV_CREATOR, config["env_id"]):
        env = _global_registry.get(ENV_CREATOR, config["env_id"])(config)
    else:
        env = gym.make(config["env_id"])

    _env, has_timelimit = env, False
    while hasattr(_env, "env"):
        if isinstance(_env, TimeLimit):
            has_timelimit = True
            break
        _env = _env.env

    if has_timelimit:
        # pylint: disable=protected-access
        _env._max_episode_steps = config["max_episode_steps"]
        # pylint: enable=protected-access
    else:
        env = TimeLimit(env, max_episode_steps=config["max_episode_steps"])

    return AddRelativeTimestep(env)


ENVS = {
    "CartPoleSwingUp": _cartpole_swingup_maker,
    "CartPoleStateless": _cartpole_stateless_maker,
    "TimeAwareEnv": _time_aware_env_maker,
}
