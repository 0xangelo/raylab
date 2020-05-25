from ray import tune


GAUSSIAN = {
    "type": "MaxEntModelBased",
    "torch_script": False,
    "model": {
        "residual": True,
        "encoder": {"units": (256, 256), "activation": "ReLU"},
    },
    "actor": {
        "input_dependent_scale": True,
        "encoder": {"units": (256, 256), "activation": "ReLU"},
    },
    "critic": {
        "encoder": {"units": (256, 256), "activation": "ReLU"},
        "target_vf": True,
    },
}

FLOW = {
    "type": "NFMBRL",
    "model": {
        "residual": True,
        "conditional_prior": True,
        "input_encoder": {"units": (128, 128), "activation": "ReLU"},
        "num_flows": 4,
        "conditional_flow": False,
        "flow": {
            "type": "AffineCouplingTransform",
            "transform_net": {"type": "MLP", "num_blocks": 0},
        },
    },
    "actor": {
        "conditional_prior": True,
        "obs_encoder": {"units": (128, 128), "activation": "ReLU"},
        "num_flows": 4,
        "conditional_flow": False,
        "flow": {
            "type": "AffineCouplingTransform",
            "transform_net": {"type": "MLP", "num_blocks": 0},
        },
    },
    "critic": {
        "encoder": {"units": (256, 256), "activation": "ReLU"},
        "target_vf": True,
    },
}


def get_config():
    return {
        # === Environment ===
        "env": "ReacherBulletEnv-v0",
        "env_config": {"max_episode_steps": 150, "time_aware": True},
        # === Optimization ===
        # PyTorch optimizers to use
        "torch_optimizer": {
            "model": {"type": "Adam", "lr": 3e-4},
            "actor": {"type": "Adam", "lr": 3e-4},
            "critic": {"type": "Adam", "lr": 3e-4},
            "alpha": {"type": "Adam", "lr": 3e-4},
        },
        # === Replay Buffer ===
        "buffer_size": 200000,
        # === Network ===
        # Size and activation of the fully connected networks computing the logits
        # for the policy, value function and model. No layers means the component is
        # linear in states and/or actions.
        "module": tune.grid_search([GAUSSIAN, FLOW]),
        # === Trainer ===
        "train_batch_size": 256,
        "timesteps_per_iteration": 1000,
        # === Exploration Settings ===
        "exploration_config": {"pure_exploration_steps": 1500},
        # === Evaluation ===
        # Evaluate with every `evaluation_interval` training iterations.
        # The evaluation stats will be reported under the "evaluation" metric key.
        "evaluation_interval": 5,
    }
