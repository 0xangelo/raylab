"""Tune experiment configuration for SVG(1) on CartPoleSwingUp.

This can be run from the command line by executing
`python scripts/tune_experiment.py 'SVG(1)' --local-dir <experiment dir>
    --config examples/svg_one_cartpole_defaults.py --stop timesteps_total 100000`
"""
from ray import tune


def get_config():
    return {
        # === Environment ===
        "env": "InvertedPendulum-v2",
        "env_config": {"max_episode_steps": 200, "time_aware": True},
        # === Replay Buffer ===
        "buffer_size": int(1e5),
        # === Optimization ===
        # Name of Pytorch optimizer class for paremetrized policy
        "optimizer": "Adam",
        # Keyword arguments to be passed to the on-policy optimizer
        "optimizer_options": {
            "model": {"lr": 1e-3},
            "value": {"lr": 1e-3},
            "policy": {"lr": 1e-3},
        },
        # Clip gradient norms by this value
        "max_grad_norm": 1e3,
        # === Regularization ===
        "kl_schedule": {
            "initial_coeff": 0.0,
            "desired_kl": 0.01,
            "adaptation_coeff": 1.01,
            "threshold": 1.0,
        },
        # === Network ===
        # Size and activation of the fully connected networks computing the logits
        # for the policy, value function and model. No layers means the component is
        # linear in states and/or actions.
        "module": {
            "actor": {
                "encoder": {
                    "units": (100, 100),
                    "activation": "Tanh",
                    "initializer_options": {"name": "orthogonal"},
                },
                "input_dependent_scale": False,
            },
            "critic": {
                "target_vf": True,
                "encoder": {
                    "units": (200, 100),
                    "activation": "ELU",
                    "initializer_options": {"name": "orthogonal"},
                },
            },
            "model": {
                "residual": True,
                "input_dependent_scale": False,
                "encoder": {
                    "units": (20, 20),
                    "activation": "Tanh",
                    "delay_action": True,
                    "initializer_options": {"name": "orthogonal"},
                },
            },
        },
        # === RolloutWorker ===
        "rollout_fragment_length": 1,
        "batch_mode": "complete_episodes",
        # === Trainer ===
        "train_batch_size": 128,
    }
