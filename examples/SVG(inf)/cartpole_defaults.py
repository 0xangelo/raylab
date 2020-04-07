"""Tune experiment configuration for SVG(inf) on CartPoleSwingUp.

This can be run from the command line by executing
`python scripts/tune_experiment.py 'SVG(inf)' --local-dir <experiment dir>
    --config examples/svg_inf_cartpole_defaults.py --stop timesteps_total 100000`
"""
import numpy as np
from ray import tune


# We can't use these callbacks since workers trying to deserialize these functions
# won't find the source. This happens because this file is imported dynamically
# and is not part of any installed library
def _on_episode_start(info):
    episode = info["episode"]
    episode.user_data["pole_angles"] = []


def _on_episode_step(info):
    episode = info["episode"]
    pole_angle = abs(episode.last_observation_for()[2])
    episode.user_data["pole_angles"].append(pole_angle)


def _on_episode_end(info):
    episode = info["episode"]
    pole_angle = np.mean(episode.user_data["pole_angles"])
    episode.custom_metrics["pole_angle"] = pole_angle


def get_config():  # pylint: disable=missing-docstring
    return {
        # === Environment ===
        "env": "CartPoleSwingUp",
        "env_config": {"max_episode_steps": 500, "time_aware": True},
        # === Replay Buffer ===
        "buffer_size": int(1e6),
        # === Optimization ===
        # Name of Pytorch optimizer class for paremetrized policy
        "on_policy_optimizer": "Adam",
        # Keyword arguments to be passed to the on-policy optimizer
        "on_policy_optimizer_options": {"lr": 3e-4},
        # Clip gradient norms by this value
        "max_grad_norm": 1e3,
        # === Regularization ===
        "kl_schedule": {
            "initial_coeff": tune.grid_search([0.0, 0.2]),
            "desired_kl": 0.01,
            "adaptation_coeff": 1.01,
            "threshold": 1.0,
        },
        # === Network ===
        # Size and activation of the fully connected networks computing the logits
        # for the policy, value function and model. No layers means the component is
        # linear in states and/or actions.
        "module": {
            "policy": {"input_dependent_scale": True},
            "model": {"delay_action": True},
        },
        # === RolloutWorker ===
        "rollout_fragment_length": 1,
        "batch_mode": "complete_episodes",
        # === Trainer ===
        "train_batch_size": 100,
        # === Debugging ===
        # Set the ray.rllib.* log level for the agent process and its workers.
        # Should be one of DEBUG, INFO, WARN, or ERROR. The DEBUG level will also
        # periodically print out summaries of relevant internal dataflow (this is
        # also printed out once at startup at the INFO level).
        "log_level": "WARN",
    }
