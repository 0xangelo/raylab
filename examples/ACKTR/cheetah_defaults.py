from ray import tune


def get_config():
    return {
        "env": "HalfCheetahBulletEnv-v0",
        "env_config": {"max_episode_steps": 1000, "time_aware": False},
        # Number of actions to sample per state for Fisher matrix approximation
        "logp_samples": 1,
        # For GAE(\gamma, \lambda)
        "gamma": 0.99,
        "lambda": 0.96,
        # Whether to use Generalized Advantage Estimation
        "use_gae": True,
        # Value function iterations per actor step
        "vf_iters": 20,
        # PyTorch optimizers to use
        "torch_optimizer": {
            "actor": {
                "type": "KFAC",
                "eps": 1e-3,
                "sua": False,
                "pi": True,
                "update_freq": 1,
                "alpha": 0.95,
                "kl_clip": 1e-2,
                "eta": 1.0,
                "lr": 1.0,
            },
            "critic": {
                "type": "KFAC",
                "eps": 1e-3,
                "sua": False,
                "pi": True,
                "update_freq": 1,
                "alpha": 0.95,
                "kl_clip": 1e-2,
                "eta": 1.0,
                "lr": 1.0,
            },
        },
        # Whether to use a line search to calculate policy update.
        # Effectively turns ACKTR into Natural PG when turned off.
        "line_search": True,
        "line_search_options": {
            "accept_ratio": 0.1,
            "backtrack_ratio": 0.8,
            "max_backtracks": 15,
            "atol": 1e-7,
        },
        # === RolloutWorker ===
        "num_workers": 0,
        "num_envs_per_worker": 16,
        "rollout_fragment_length": 125,
        "batch_mode": "truncate_episodes",
        "timesteps_per_iteration": 2000,
        # === Network ===
        # Size and activation of the fully connected networks computing the logits
        # for the policy and value function. No layers means the component is
        # linear in states or actions.
        "module": {
            "name": "OnPolicyActorCritic",
            "torch_script": False,
            "actor": {
                "encoder": {
                    "units": (64, 32),
                    "activation": "ELU",
                    "initializer_options": {"name": "orthogonal"},
                },
                "input_dependent_scale": False,
            },
            "critic": {
                "encoder": {
                    "units": (64, 32),
                    "activation": "ELU",
                    "initializer_options": {"name": "orthogonal"},
                },
            },
        },
    }
