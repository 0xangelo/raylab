from ray.tune import register_env, register_trainable


def _register_all_algorithms():
    from ray.rllib.contrib.registry import CONTRIBUTED_ALGORITHMS
    from raylab.algorithms.registry import LOCAL_ALGORITHMS

    CONTRIBUTED_ALGORITHMS.update(LOCAL_ALGORITHMS)
    for key in LOCAL_ALGORITHMS:
        from ray.rllib.agents.registry import get_agent_class

        register_trainable(key, get_agent_class(key))


def _register_all_envs():
    from raylab.envs.registry import LOCAL_ENVS

    for name, maker in LOCAL_ENVS.items():
        register_env(name, maker)


_register_all_algorithms()
_register_all_envs()
