from ray import tune
from ray.rllib.utils import merge_dicts

from ib_defaults import get_config as base_config


def get_config():
    return merge_dicts(
        base_config(),
        {
            # === Environment ===
            "env_config": {
                "setpoint": 50,
                "miscalibration": tune.grid_search([True, False]),
                "max_episode_steps": 1000,
            },
            # === MAPO model training ===
            "model_rollout_len": 1,
            # === Debugging ===
            # Whether to use the environment's true model to sample states
            "true_model": True,
            # === Replay Buffer ===
            "buffer_size": int(1e5),
            # === Network ===
            # Size and activation of the fully connected networks computing the logits
            # for the policy and action-value function. No layers means the component is
            # linear in states and/or actions.
            "module": {
                "actor": {"encoder": {"units": (256, 256)}},
                "critic": {"encoder": {"units": (256, 256)}},
                "model": {"encoder": {"units": (256, 256)}},
            },
            # === Trainer ===
            "train_batch_size": 256,
            "timesteps_per_iteration": 1000,
            # === Exploration Settings ===
            "exploration_config": {"pure_exploration_steps": 2000},
            # === Evaluation ===
            "evaluation_interval": 1,
            "evaluation_num_episodes": 5,
        },
    )
