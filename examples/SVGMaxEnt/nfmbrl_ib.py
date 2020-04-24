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
        "env": "IndustrialBenchmark",
        "env_config": {
            "max_episode_steps": 1000,
            "time_aware": False,
            "setpoint": 50,
            "reward_type": "classic",
            "observation": "visible",
        },
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
        # "module": GAUSSIAN,
        # "module": FLOW,
        # === Trainer ===
        "train_batch_size": 256,
        "timesteps_per_iteration": 1000,
        # === Exploration Settings ===
        # Default exploration behavior, iff `explore`=None is passed into
        # compute_action(s).
        # Set to False for no exploration behavior (e.g., for evaluation).
        "explore": True,
        # Provide a dict specifying the Exploration object's config.
        "exploration_config": {
            # The Exploration class to use. In the simplest case, this is the name
            # (str) of any class present in the `rllib.utils.exploration` package.
            # You can also provide the python class directly or the full location
            # of your class (e.g. "ray.rllib.utils.exploration.epsilon_greedy.
            # EpsilonGreedy").
            "type": "raylab.utils.exploration.StochasticActor",
            # Options for parameter noise exploration
            # Until this many timesteps have elapsed, the agent's policy will be
            # ignored & it will instead take uniform random actions. Can be used in
            # conjunction with learning_starts (which controls when the first
            # optimization step happens) to decrease dependence of exploration &
            # optimization on initial policy parameters. Note that this will be
            # disabled when the action noise scale is set to 0 (e.g during evaluation).
            "pure_exploration_steps": 10000,
        },
        # === Evaluation ===
        # Evaluate with every `evaluation_interval` training iterations.
        # The evaluation stats will be reported under the "evaluation" metric key.
        "evaluation_interval": 5,
        # Extra arguments to pass to evaluation workers.
        # Typical usage is to pass extra args to evaluation env creator
        # and to disable exploration by computing deterministic actions
        "evaluation_config": {"explore": False},
    }
