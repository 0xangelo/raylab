from ray import tune


def get_config():
    return {
        # === Environment ===
        "env": "HalfCheetah-v3",
        "env_config": {"max_episode_steps": 1000, "time_aware": False},
        # Trust region constraint
        "delta": 0.01,
        # Number of actions to sample per state for Fisher vector product approximation
        "fvp_samples": 20,
        # For GAE(\gamma, \lambda)
        "gamma": 0.995,
        "lambda": 0.96,
        # Number of iterations to fit value function
        "val_iters": 40,
        # Options for critic optimizer
        "torch_optimizer": {"type": "Adam", "lr": 1e-2},
        # === RolloutWorker ===
        "num_workers": 3,
        "num_envs_per_worker": 6,
        "rollout_fragment_length": 400,
        "batch_mode": "truncate_episodes",
        "timesteps_per_iteration": 7200,
        # === Network ===
        "module": {
            "actor": {
                "encoder": {
                    "units": (32, 32),
                    "activation": "ELU",
                    "initializer_options": {"name": "orthogonal"},
                },
            },
            "critic": {
                "encoder": {
                    "units": (32, 32),
                    "activation": "ELU",
                    "initializer_options": {"name": "orthogonal"},
                },
            },
        },
    }
