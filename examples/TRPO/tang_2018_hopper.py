from ray import tune


def get_config():
    return {
        # === Environment ===
        "env": "Hopper-v2",
        "env_config": {"max_episode_steps": 1000, "time_aware": False},
        # Trust region constraint
        "delta": 0.01,
        # Number of actions to sample per state for Fisher vector product approximation
        "fvp_samples": 10,
        # For GAE(\gamma, \lambda)
        "gamma": 0.99,
        "lambda": 0.98,
        # Number of iterations to fit value function
        "val_iters": 5,
        # Options for critic optimizer
        "optimizer": {"type": "Adam", "lr": 1e-3},
        # Configuration for Conjugate Gradient
        "cg_iters": 10,
        "cg_damping": 1e-3,
        # Whether to use a line search to calculate policy update.
        # Effectively turns TRPO into Natural PG when turned off.
        "line_search": True,
        "line_search_options": {
            "accept_ratio": 0.0,
            "backtrack_ratio": 0.5,
            "max_backtracks": 10,
            "atol": 1e-7,
        },
        # === RolloutWorker ===
        "num_workers": 0,
        "num_envs_per_worker": 1,
        "rollout_fragment_length": 1024,
        "batch_mode": "truncate_episodes",
        "timesteps_per_iteration": 1024,
        # === Network ===
        "module": {
            "name": "TRPOTang2018",
            "torch_script": True,
            "actor": {"num_flows": 4, "hidden_size": 3},
            "critic": {
                "encoder": {
                    "units": (32, 32),
                    "activation": "Tanh",
                    "initializer_options": {"name": "normal", "std": 1.0},
                },
            },
        },
    }
