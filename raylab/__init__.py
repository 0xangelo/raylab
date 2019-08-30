"""RAYLAB: Extensions and custom algorithms in RLlib."""


def register_all_agents():
    """Register all trainer names in Tune."""
    from ray.tune import register_trainable
    from raylab.algorithms.registry import ALGORITHMS

    for name, trainer_import in ALGORITHMS.items():
        register_trainable(name, trainer_import())


def register_all_environments():
    """Register all custom environments in Tune."""
    from ray.tune import register_env
    from raylab.envs.registry import ENVS

    for name, env_import in ENVS.items():
        register_env(name, env_import)
