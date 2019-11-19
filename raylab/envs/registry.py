"""Registry of custom Gym environments."""
import inspect

from .utils import wrap_if_needed


@wrap_if_needed
def _cartpole_swingup_maker(_):
    from raylab.envs.cartpole_swingup import CartPoleSwingUpEnv

    return CartPoleSwingUpEnv()


@wrap_if_needed
def _cartpole_stateless_maker(_):
    from gym.envs.classic_control.cartpole import CartPoleEnv
    from raylab.envs.cartpole_stateless import CartPoleStatelessWrapper

    return CartPoleStatelessWrapper(CartPoleEnv())


@wrap_if_needed
def _navigation_maker(config):
    from raylab.envs.navigation import NavigationEnv

    return NavigationEnv(config)


@wrap_if_needed
def _reservoir_maker(config):
    from raylab.envs.reservoir import ReservoirEnv

    return ReservoirEnv(config)


@wrap_if_needed
def _hvac_maker(config):
    from raylab.envs.hvac import HVACEnv

    return HVACEnv(config)


@wrap_if_needed
def _mujoco_reacher_maker(_):
    from raylab.envs.reacher import ReacherEnv

    return ReacherEnv()


@wrap_if_needed
def _mujoco_half_cheetah_maker(_):
    from raylab.envs.half_cheetah import HalfCheetahEnv

    return HalfCheetahEnv()


@wrap_if_needed
def _industrial_benchmark_maker(config):
    from raylab.envs.industrial_benchmark.openai_ib import IBEnv

    return IBEnv(
        **{k: config[k] for k in inspect.signature(IBEnv).parameters if k in config}
    )


ENVS = {
    "CartPoleSwingUp": _cartpole_swingup_maker,
    "CartPoleStateless": _cartpole_stateless_maker,
    "Navigation": _navigation_maker,
    "Reservoir": _reservoir_maker,
    "HVAC": _hvac_maker,
    "MujocoReacher": _mujoco_reacher_maker,
    "MujocoHalfCheetah": _mujoco_half_cheetah_maker,
    "IndustrialBenchmark": _industrial_benchmark_maker,
}
