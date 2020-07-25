"""Functions for Trainer setups as in RLlib."""
# pylint:disable=missing-function-docstring,logging-format-interpolation
import copy
import logging
from typing import Any
from typing import Callable
from typing import Optional

from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.trainer import Trainer
from ray.rllib.env.normalize_actions import NormalizeActionWrapper
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.deprecation import DEPRECATED_VALUE
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.from_config import from_config
from ray.tune.registry import _global_registry
from ray.tune.registry import ENV_CREATOR
from ray.tune.resources import Resources

logger = logging.getLogger(__name__)
tf = try_import_tf()


# ==============================================================================
# Trainer setup
# ==============================================================================
def make_env_creator(env_id: Optional[str], config: dict) -> Callable[[dict], Any]:
    if env_id:
        config["env"] = env_id
        # An already registered env.
        if _global_registry.contains(ENV_CREATOR, env_id):
            env_creator = _global_registry.get(ENV_CREATOR, env_id)
        # A class specifier.
        elif "." in env_id:

            def env_creator(env_config):
                return from_config(env_id, env_config)

        # Try gym.
        else:
            import gym  # soft dependency

            def env_creator(_):
                return gym.make(env_id)

    else:

        def env_creator(_):
            return

    if config["normalize_actions"]:
        env_creator = normalize_env_creator(env_creator)
    return env_creator


def check_and_resolve_framework_settings(config: dict):
    # Check and resolve DL framework settings.
    if "use_pytorch" in config and config["use_pytorch"] != DEPRECATED_VALUE:
        deprecation_warning("use_pytorch", "framework=torch", error=False)
        if config["use_pytorch"]:
            config["framework"] = "torch"
        config.pop("use_pytorch")
    if "eager" in config and config["eager"] != DEPRECATED_VALUE:
        deprecation_warning("eager", "framework=tfe", error=False)
        if config["eager"]:
            config["framework"] = "tfe"
        config.pop("eager")

    # Enable eager/tracing support.
    if tf and config["framework"] == "tfe":
        if not tf.executing_eagerly():
            tf.enable_eager_execution()
        logger.info(
            "Executing eagerly, with eager_tracing={}".format(config["eager_tracing"])
        )
    if tf and not tf.executing_eagerly() and config["framework"] != "torch":
        logger.info(
            "Tip: set framework=tfe or the --eager flag to enable "
            "TensorFlow eager execution"
        )


def normalize_env_creator(env_creator: callable) -> callable:
    inner = env_creator

    def normalize(env):
        import gym  # soft dependency

        if not isinstance(env, gym.Env):
            raise ValueError(
                "Cannot apply NormalizeActionActionWrapper to env of "
                "type {}, which does not subclass gym.Env.".format(type(env)),
            )
        return NormalizeActionWrapper(env)

    return lambda env_config: normalize(inner(env_config))


def validate_callbacks(config: dict) -> DefaultCallbacks:
    if not callable(config["callbacks"]):
        raise ValueError(
            "`callbacks` must be a callable method that "
            "returns a subclass of DefaultCallbacks, got {}".format(config["callbacks"])
        )
    return config["callbacks"]()


def set_rllib_log_level(config: dict):
    log_level = config.get("log_level")
    if log_level in ["WARN", "ERROR"]:
        logger.info(
            "Current log_level is {}. For more information, "
            "set 'log_level': 'INFO' / 'DEBUG' or use the -v and "
            "-vv flags.".format(log_level)
        )
    if config.get("log_level"):
        logging.getLogger("ray.rllib").setLevel(config["log_level"])


def setup_evaluation_config(config: dict) -> dict:
    # Update env_config with evaluation settings:
    extra_config = copy.deepcopy(config["evaluation_config"])
    # Assert that user has not unset "in_evaluation".
    assert "in_evaluation" not in extra_config or extra_config["in_evaluation"] is True
    extra_config.update(
        {
            "batch_mode": "complete_episodes",
            "rollout_fragment_length": 1,
            "in_evaluation": True,
        }
    )
    logger.debug("using evaluation_config: {}".format(extra_config))
    return merge_dicts(config, extra_config)


# ==============================================================================
# Trainer resource requests
# ==============================================================================
def default_resource_request(cls, config: dict) -> Resources:
    cf = dict(cls.options.defaults, **config)  # pylint:disable=invalid-name
    Trainer._validate_config(cf)  # pylint:disable=protected-access
    num_workers = cf["num_workers"] + cf["evaluation_num_workers"]
    return Resources(
        cpu=cf["num_cpus_for_driver"],
        gpu=cf["num_gpus"],
        memory=cf["memory"],
        object_store_memory=cf["object_store_memory"],
        extra_cpu=cf["num_cpus_per_worker"] * num_workers,
        extra_gpu=cf["num_gpus_per_worker"] * num_workers,
        extra_memory=cf["memory_per_worker"] * num_workers,
        extra_object_store_memory=cf["object_store_memory_per_worker"] * num_workers,
    )
