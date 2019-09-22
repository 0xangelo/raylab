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
    from raylab.envs.time_aware_env import AddRelativeTimestep

    env = gym.make(config["env_id"])
    has_timelimit = False
    while hasattr(env, "env"):
        if isinstance(env, TimeLimit):
            has_timelimit = True
            break

    if has_timelimit:
        env.spec.max_episode_steps = config["max_episode_steps"]
        # pylint: disable=protected-access
        env._max_episode_steps = config["max_episode_steps"]
        # pylint: enable=protected-access
    else:
        env = TimeLimit(env, max_episode_steps=config["max_episode_steps"])

    return AddRelativeTimestep(env)


ENVS = {
    "CartPoleSwingUp": _cartpole_swingup_maker,
    "CartPoleStateless": _cartpole_stateless_maker,
    "TimeAwareEnv": _time_aware_env_maker,
}
