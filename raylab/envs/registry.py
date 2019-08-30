def cartpole_swingup_maker(config):
    from raylab.envs.cartpole_swingup import CartPoleSwingUpEnv

    return CartPoleSwingUpEnv(config)


def cartpole_stateless_maker(_):
    from gym.envs.classic_control.cartpole import CartPoleEnv
    from raylab.envs.cartpole_stateless import CartPoleStatelessWrapper

    return CartPoleStatelessWrapper(CartPoleEnv())


LOCAL_ENVS = {
    "CartPoleSwingUp": cartpole_swingup_maker,
    "CartPoleStateless": cartpole_stateless_maker,
}
