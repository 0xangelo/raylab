# pylint:disable=import-outside-toplevel
"""Registry of custom Gym environments."""
import importlib

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


def register_external_library_environments(library_name):
    """Conveniency function for adding external environments to the global registry."""
    if importlib.util.find_spec(library_name) is None:
        return

    importlib.import_module(library_name)
    new_ids = filtered_gym_env_ids() - IDS
    for name in new_ids:

        @wrap_if_needed
        def _env_maker(config, env_id=name):
            importlib.import_module(library_name)

            return gym.make(env_id, **config)

        ENVS[name] = _env_maker
    IDS.update(new_ids)


@wrap_if_needed
def _cartpole_stateless_maker(_):
    from raylab.envs.environments.cartpole_stateless import CartPoleStateless

    return CartPoleStateless()


@wrap_if_needed
def _navigation_maker(config):
    from raylab.envs.environments.navigation import NavigationEnv

    return NavigationEnv(config)


@wrap_if_needed
def _reservoir_maker(config):
    from raylab.envs.environments.reservoir import ReservoirEnv

    return ReservoirEnv(config)


@wrap_if_needed
def _hvac_maker(config):
    from raylab.envs.environments.hvac import HVACEnv

    return HVACEnv(config)


ENVS.update(
    {
        "CartPoleStateless": _cartpole_stateless_maker,
        "Navigation": _navigation_maker,
        "Reservoir": _reservoir_maker,
        "HVAC": _hvac_maker,
    }
)


register_external_library_environments("gym_cartpole_swingup")
register_external_library_environments("gym_industrial")
register_external_library_environments("pybullet_envs")
