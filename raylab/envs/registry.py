"""Registry of custom Gym environments."""


def _cartpole_swingup_maker(_):
    from gym_cartpole_swingup.envs import CartPoleSwingUpEnv

    return CartPoleSwingUpEnv()


def _cartpole_stateless_maker(_):
    from gym.envs.classic_control.cartpole import CartPoleEnv
    from raylab.envs.cartpole_stateless import CartPoleStatelessWrapper

    return CartPoleStatelessWrapper(CartPoleEnv())


ENVS = {
    "CartPoleSwingUp": _cartpole_swingup_maker,
    "CartPoleStateless": _cartpole_stateless_maker,
}
