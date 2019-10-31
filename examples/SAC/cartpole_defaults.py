"""Tune experiment configuration to test SAC in CartPoleSwingUp.

This can be run from the command line by executing
`raylab experiment SAC --config examples/naf_exploration_experiment.py -s timesteps_total 100000`
"""
from ray import tune


def get_config():
    return {
        # === Environment ===
        "env": "CartPoleSwingUp",
        "env_config": {"max_episode_steps": 500, "time_aware": False},
        # === Replay Buffer ===
        "buffer_size": int(1e5),
        # === Optimization ===
        # PyTorch optimizer to use for policy
        "policy_optimizer": {"name": "Adam", "options": {"lr": 3e-4}},
        # PyTorch optimizer to use for critic
        "critic_optimizer": {"name": "Adam", "options": {"lr": 3e-4}},
        # PyTorch optimizer to use for entropy coefficient
        "alpha_optimizer": {"name": "Adam", "options": {"lr": 3e-4}},
        # === Network ===
        # Size and activation of the fully connected networks computing the logits
        # for the policy and action-value function. No layers means the component is
        # linear in states and/or actions.
        "module": {
            "policy": {
                "units": (128, 128),
                "activation": "ReLU",
                "initializer_options": {"name": "xavier_uniform"},
                "input_dependent_scale": True,
            },
            "critic": {
                "units": (128, 128),
                "activation": "ReLU",
                "initializer_options": {"name": "xavier_uniform"},
                "delay_action": True,
            },
        },
        # === Exploration ===
        "pure_exploration_steps": 5000,
        # === Trainer ===
        "train_batch_size": 128,
        "timesteps_per_iteration": 1000,
        # === Evaluation ===
        # Evaluate with every `evaluation_interval` training iterations.
        # The evaluation stats will be reported under the "evaluation" metric key.
        "evaluation_interval": 5,
        # Number of episodes to run per evaluation period.
        "evaluation_num_episodes": 5,
        # === Debugging ===
        # Set the ray.rllib.* log level for the agent process and its workers.
        # Should be one of DEBUG, INFO, WARN, or ERROR. The DEBUG level will also
        # periodically print out summaries of relevant internal dataflow (this is
        # also printed out once at startup at the INFO level).
        "log_level": "WARN",
    }
