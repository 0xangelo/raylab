from ray import tune


def get_config():
    return {
        # === Environment ===
        "env": "ReacherBulletEnv-v0",
        "env_config": {"max_episode_steps": 150, "time_aware": True},
        # === Replay Buffer ===
        "buffer_size": int(2e5),
        # === Optimization ===
        # PyTorch optimizers to use
        "optimizer": {
            "actor": {"type": "Adam", "lr": 3e-4},
            "critics": {"type": "Adam", "lr": 3e-4},
            "alpha": {"type": "Adam", "lr": 3e-4},
        },
        # === Network ===
        # Size and activation of the fully connected networks computing the logits
        # for the policy and action-value function. No layers means the component is
        # linear in states and/or actions.
        "module": {
            "actor": {"encoder": {"units": (128, 128)}},
            "critic": {"encoder": {"units": (128, 128)}},
        },
        # === Trainer ===
        "train_batch_size": 128,
        "timesteps_per_iteration": 1000,
        # === Exploration Settings ===
        "exploration_config": {"pure_exploration_steps": 200},
        # === Evaluation ===
        # Evaluate with every `evaluation_interval` training iterations.
        # The evaluation stats will be reported under the "evaluation" metric key.
        "evaluation_interval": 5,
    }
