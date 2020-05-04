# pylint:disable=import-outside-toplevel
"""Registry of custom Gym environments."""
import importlib
import inspect

import gym

from .utils import wrap_if_needed


def filtered_gym_env_ids():
    """
    Return environment ids in Gym registry for which all dependencies are installed.
    """
    specs = set(gym.envs.registry.all())

    if importlib.util.find_spec("atari_py") is None:
        specs.difference_update({s for s in specs if "atari" in s.entry_point})
    if importlib.util.find_spec("mujoco_py") is None:
        specs.difference_update({s for s in specs if "mujoco" in s.entry_point})
        specs.difference_update({s for s in specs if "robotics" in s.entry_point})
    if importlib.util.find_spec("Box2D") is None:
        specs.difference_update({s for s in specs if "box2d" in s.entry_point})

    return {s.id for s in specs}


IDS = filtered_gym_env_ids()
# kwarg trick from:
# https://github.com/satwikkansal/wtfpython#-the-sticky-output-function
ENVS = {i: wrap_if_needed(lambda config, i=i: gym.make(i, **config)) for i in IDS}


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
def _industrial_benchmark_maker(config):
    from raylab.envs.industrial_benchmark.openai_ib import IBEnv

    return IBEnv(
        **{k: config[k] for k in inspect.signature(IBEnv).parameters if k in config}
    )


ENVS.update(
    {
        "CartPoleStateless": _cartpole_stateless_maker,
        "Navigation": _navigation_maker,
        "Reservoir": _reservoir_maker,
        "HVAC": _hvac_maker,
        "IndustrialBenchmark": _industrial_benchmark_maker,
    }
)


try:
    import gym_cartpole_swingup  # pylint:disable=unused-import,import-error

    NEW_IDS = filtered_gym_env_ids() - IDS
    for id_ in NEW_IDS:

        @wrap_if_needed
        def _swingup_env_maker(config, id_=id_):
            # pylint:disable=redefined-outer-name,reimported,unused-import
            import gym_cartpole_swingup  # noqa:F811

            return gym.make(id_, **config)

        ENVS[id_] = _swingup_env_maker
    IDS.update(NEW_IDS)
except ImportError:
    pass


try:
    import pybullet_envs  # pylint:disable=unused-import,import-error

    NEW_IDS = filtered_gym_env_ids() - IDS
    for id_ in NEW_IDS:

        @wrap_if_needed
        def _pybullet_env_maker(config, id_=id_):
            # pylint:disable=redefined-outer-name,reimported,unused-import
            import pybullet_envs  # noqa:F811

            return gym.make(id_, **config)

        ENVS[id_] = _pybullet_env_maker
    IDS.update(NEW_IDS)
except ImportError:
    pass
