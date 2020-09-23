from ray import tune


def get_config():
    return {
        # === Environment ===
        "env": "Reacher-v2",
        "env_config": {"max_episode_steps": 50, "time_aware": True},
        # === Optimization ===
        # PyTorch optimizers to use
        "optimizer": {
            "model": {"type": "Adam", "lr": 3e-4},
            "actor": {"type": "Adam", "lr": 3e-4},
            "critic": {"type": "Adam", "lr": 3e-4},
            "alpha": {"type": "Adam", "lr": 3e-4},
        },
        # === Replay Buffer ===
        "buffer_size": 80000,
        # === Network ===
        # Size and activation of the fully connected networks computing the logits
        # for the policy, value function and model. No layers means the component is
        # linear in states and/or actions.
        "module": {
            "name": "MaxEntModelBased",
            "torch_script": False,
            "model": {
                "residual": True,
                "encoder": {
                    "units": tune.grid_search([(64, 64), (128, 128), (256, 256)])
                },
            },
            "actor": {"input_dependent_scale": True, "encoder": {"units": (128, 128)}},
            "critic": {"encoder": {"units": (128, 128)}, "target_vf": True},
        },
        # === Trainer ===
        "train_batch_size": 128,
        "timesteps_per_iteration": 1000,
        # === Exploration Settings ===
        "exploration_config": {"pure_exploration_steps": 500},
        # === Evaluation ===
        # Evaluate with every `evaluation_interval` training iterations.
        # The evaluation stats will be reported under the "evaluation" metric key.
        "evaluation_interval": 5,
    }
