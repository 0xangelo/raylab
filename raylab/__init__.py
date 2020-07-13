"""RAYLAB: Extensions and custom algorithms in RLlib."""
import poetry_version

__author__ = """Ângelo Gregório Lovatto"""
__email__ = "angelolovatto@gmail.com"
__version__ = poetry_version.extract(source_file=__file__)


def register_all_agents():
    """Register all trainer names in Tune."""
    from ray.tune import register_trainable
    from raylab.agents.registry import AGENTS

    for name, trainer_import in AGENTS.items():
        register_trainable(name, trainer_import())


def register_all_environments():
    """Register all custom environments in Tune."""
    from ray.tune import register_env
    from raylab.envs.registry import ENVS

    for name, env_creator in ENVS.items():
        register_env(name, env_creator)


def register_all():
    """Register all trainers and environments in Tune."""
    register_all_agents()
    register_all_environments()
