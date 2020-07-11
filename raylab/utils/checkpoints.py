"""Utilities for handling experiment results."""
import os.path as osp
import pickle
import warnings

from ray.rllib.utils import merge_dicts

from raylab.agents.registry import get_agent_cls


def get_agent_from_checkpoint(checkpoint, agent_name, env=None, **config_kwargs):
    """Instatiate and restore agent class from checkpoint."""
    config = get_config_from_checkpoint(checkpoint, **config_kwargs)
    agent_cls = get_agent_cls(agent_name)

    agent = agent_cls(env=env, config=config)
    agent.restore(checkpoint)
    return agent


def get_config_from_checkpoint(checkpoint, use_eval_config=True, config_overrides=None):
    """Find and load configuration for checkpoint file."""
    config = {}
    # Load configuration from file
    config_dir = osp.dirname(checkpoint)
    config_path = osp.join(config_dir, "params.pkl")
    if not osp.exists(config_path):
        config_path = osp.join(config_dir, "../params.pkl")
    if not osp.exists(config_path):
        raise ValueError(
            "Could not find params.pkl in either the checkpoint dir or "
            "its parent directory."
        )
    with open(config_path, "rb") as file:
        config = pickle.load(file)

    if use_eval_config:
        if "evaluation_config" not in config:
            warnings.warn("Evaluation agent requested but none in config.")
        eval_conf = config.get("evaluation_config", {})
        config = merge_dicts(config, eval_conf)

    if config_overrides:
        config = merge_dicts(config, config_overrides)
    return config
