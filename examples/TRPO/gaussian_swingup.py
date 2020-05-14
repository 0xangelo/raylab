from ray import tune


def get_config():
    return {
        # === Environment ===
        "env": "CartPoleSwingUp-v1",
        "env_config": {"max_episode_steps": 500, "time_aware": True},
        # Trust region constraint
        "delta": 0.01,
        # Number of actions to sample per state for Fisher vector product approximation
        "fvp_samples": 20,
        # For GAE(\gamma, \lambda)
        "gamma": 0.99,
        "lambda": 0.97,
        # Number of iterations to fit value function
        "val_iters": 40,
        # Learning rate for critic optimizer
        "val_lr": 1e-2,
        # === RolloutWorker ===
        "num_workers": 2,
        "num_envs_per_worker": 8,
        "rollout_fragment_length": 400,
        "batch_mode": "truncate_episodes",
        "timesteps_per_iteration": 6400,
        # === Network ===
        "module": {
            "actor": {
                "units": (32, 32),
                "activation": "ELU",
                "initializer_options": {"name": "orthogonal"},
            },
            "critic": {
                "units": (32, 32),
                "activation": "ELU",
                "initializer_options": {"name": "orthogonal"},
            },
        },
    }
