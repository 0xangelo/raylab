"""Tune experiment configuration for SVG(1) on MujocoReacher."""
import numpy as np
from ray import tune  # pylint: disable=unused-import


def get_config():  # pylint: disable=missing-docstring
    return {
        # === Environment ===
        "env": "MujocoReacher",
        "env_config": {"max_episode_steps": 50, "time_aware": True},
        # === Replay Buffer ===
        "buffer_size": int(2e5),
        # === Optimization ===
        # Name of Pytorch optimizer class for paremetrized policy
        "torch_optimizer": "Adam",
        # Keyword arguments to be passed to the on-policy optimizer
        "torch_optimizer_options": {
            "model": {"lr": 1e-3},
            "value": {"lr": 1e-3},
            "policy": {"lr": 1e-3},
        },
        # Clip gradient norms by this value
        "max_grad_norm": 1e3,
        # === Regularization ===
        "kl_schedule": {
            "initial_coeff": 0.5,
            "desired_kl": 0.01,
            "adaptation_coeff": 1.01,
            "threshold": 1.0,
        },
        # === Network ===
        # Size and activation of the fully connected networks computing the logits
        # for the policy, value function and model. No layers means the component is
        # linear in states and/or actions.
        "module": {
            "policy": {
                "layers": (100, 100),
                "activation": "Tanh",
                "input_dependent_scale": True,
                "initializer_options": {"name": "orthogonal"},
            },
            "value": {
                "layers": (400, 200),
                "activation": "Tanh",
                "initializer_options": {"name": "orthogonal", "gain": np.sqrt(2)},
            },
            "model": {
                "layers": (40, 40),
                "activation": "Tanh",
                "delay_action": True,
                "initializer_options": {"name": "orthogonal"},
            },
        },
        # === RolloutWorker ===
        "sample_batch_size": 1,
        "batch_mode": "complete_episodes",
        "timesteps_per_iteration": 1000,
        # === Trainer ===
        "train_batch_size": 128,
        # === Evaluation ===
        "evaluation_interval": 1,
        # Extra arguments to pass to evaluation workers.
        # Typical usage is to pass extra args to evaluation env creator
        # and to disable exploration by computing deterministic actions
        "evaluation_config": {"mean_action_only": True, "pure_exploration_steps": 0},
        # === Debugging ===
        # Set the ray.rllib.* log level for the agent process and its workers.
        # Should be one of DEBUG, INFO, WARN, or ERROR. The DEBUG level will also
        # periodically print out summaries of relevant internal dataflow (this is
        # also printed out once at startup at the INFO level).
        "log_level": "WARN",
    }
