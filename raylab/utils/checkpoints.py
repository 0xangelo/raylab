"""Utilities for handling experiment results."""
import os.path as osp
import pickle
from ray.tune.registry import TRAINABLE_CLASS, _global_registry
from ray.rllib.utils import merge_dicts


def get_agent(checkpoint, algo, env):
    """Instatiate and restore agent class from checkpoint."""
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

    if "evaluation_config" in config:
        eval_conf = config["evaluation_config"]
        config = merge_dicts(config, eval_conf)

    agent_cls = _global_registry.get(TRAINABLE_CLASS, algo)
    agent = agent_cls(env=env, config=config)
    agent.restore(checkpoint)
    return agent
