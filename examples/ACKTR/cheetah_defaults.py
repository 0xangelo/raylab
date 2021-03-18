"""Remember to install PyBullet first:
```
pip install pybullet
```
"""


def get_config():
    from raylab.policy.modules.trpo import TRPOSpec

    module_spec = TRPOSpec()
    module_spec.actor.encoder.units = (64, 32)
    module_spec.actor.encoder.activation = "ELU"
    module_spec.actor.initializer = {"name": "orthogonal"}
    module_spec.actor.input_dependent_scale = False

    module_spec.critic.units = (64, 32)
    module_spec.critic.activation = "ELU"

    return {
        "env": "HalfCheetahBulletEnv-v0",
        "env_config": {"max_episode_steps": 1000, "time_aware": False},
        "policy": {
            # Number of actions to sample per state for Fisher matrix approximation
            "fvp_samples": 10,
            # For GAE(\gamma, \lambda)
            "gamma": 0.99,
            "lambda": 0.96,
            # Whether to use Generalized Advantage Estimation
            "use_gae": True,
            # Value function iterations per actor step
            "val_iters": 20,
            # PyTorch optimizers to use
            "optimizer": {
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
            # === Network ===
            # Size and activation of the fully connected networks computing the logits
            # for the policy and value function. No layers means the component is
            # linear in states or actions.
            "module": {"type": "TRPO", "torch_script": False, **module_spec.to_dict()},
        },
        # === RolloutWorker ===
        "num_workers": 0,
        "num_envs_per_worker": 16,
        "rollout_fragment_length": 125,
        "batch_mode": "truncate_episodes",
        "timesteps_per_iteration": 2000,
    }
