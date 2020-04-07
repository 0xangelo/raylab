# pylint:disable=missing-docstring,unused-import
import numpy as np
from ray import tune


def get_config():
    return {
        # === Environment ===
        "env": "HalfCheetah-v3",
        "env_config": {"max_episode_steps": 400, "time_aware": False},
        # Trust region constraint
        "delta": 0.01,
        # Number of actions to sample per state for Fisher vector product approximation
        "fvp_samples": 10,
        # For GAE(\gamma, \lambda)
        "gamma": 0.995,
        "lambda": 0.96,
        # Number of iterations to fit value function
        "val_iters": 40,
        # Learning rate for critic optimizer
        "val_lr": 1e-2,
        # Whether to use Generalized Advantage Estimation
        "use_gae": True,
        # Whether to use a line search to calculate policy update.
        # Effectively turns TRPO into Natural PG when turned off.
        "line_search": True,
        # === RolloutWorker ===
        "num_workers": 0,
        "num_envs_per_worker": 12,
        "rollout_fragment_length": 400,
        "batch_mode": "truncate_episodes",
        "timesteps_per_iteration": 4800,
        # === Network ===
        # Size and activation of the fully connected networks computing the logits
        # for the policy and value function. No layers means the component is
        # linear in states or actions.
        "module": {
            "name": "TRPORealNVP",
            "torch_script": False,
            "actor": {
                "units": (32,),
                "activation": "ELU",
                "initializer_options": {"name": "orthogonal"},
                "num_flows": 4,
                "flow": {
                    "units": (24,) * 2,
                    "activation": "ELU",
                    "initializer_options": {"name": "orthogonal"},
                },
            },
            "critic": {
                "units": (32, 32),
                "activation": "ELU",
                "initializer_options": {"name": "orthogonal"},
            },
        },
    }
