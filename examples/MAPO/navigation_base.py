from ray import tune


def get_config():
    return {
        # === Environment ===
        "env": "Navigation",
        # === Replay Buffer ===
        "buffer_size": int(1e4),
        # === Twin Delayed DDPG (TD3) tricks ===
        # Clipped Double Q-Learning: use the minimun of two target Q functions
        # as the next action-value in the target for fitted Q iteration
        "clipped_double_q": True,
        # === Optimization ===
        # PyTorch optimizers to use
        "torch_optimizer": {
            "model": {"type": "Adam", "lr": 3e-4},
            "actor": {"type": "Adam", "lr": 3e-4},
            "critics": {"type": "Adam", "lr": 3e-4},
        },
        # Interpolation factor in polyak averaging for target networks.
        "polyak": 0.995,
        # === Network ===
        # Size and activation of the fully connected networks computing the logits
        # for the policy and action-value function. No layers means the component is
        # linear in states and/or actions.
        "module": {
            "type": "MAPOModule",
            "actor": {
                "smooth_target_policy": True,
                "target_gaussian_sigma": 0.3,
                "encoder": {
                    "units": (64, 64),
                    "activation": "ReLU",
                    "initializer_options": {"name": "xavier_uniform"},
                },
            },
            "critic": {
                "encoder": {
                    "units": (64, 64),
                    "activation": "ReLU",
                    "delay_action": True,
                    "initializer_options": {"name": "xavier_uniform"},
                },
            },
            "model": {
                "input_dependent_scale": False,
                "encoder": {
                    "units": (64, 64),
                    "activation": "ReLU",
                    "initializer_options": {"name": "xavier_uniform"},
                    "delay_action": True,
                },
            },
        },
        # === RolloutWorker ===
        "rollout_fragment_length": 1,
        "batch_mode": "complete_episodes",
        # === Trainer ===
        "train_batch_size": 32,
        "timesteps_per_iteration": 200,
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
            "type": "raylab.utils.exploration.GaussianNoise",
            # Options for Gaussian noise exploration
            "noise_stddev": 0.3,
            # Until this many timesteps have elapsed, the agent's policy will be
            # ignored & it will instead take uniform random actions. Can be used in
            # conjunction with learning_starts (which controls when the first
            # optimization step happens) to decrease dependence of exploration &
            # optimization on initial policy parameters. Note that this will be
            # disabled when the action noise scale is set to 0 (e.g during evaluation).
            "pure_exploration_steps": 200,
        },
        # === Evaluation ===
        "evaluation_interval": 5,
        "evaluation_num_episodes": 5,
        # Extra arguments to pass to evaluation workers.
        # Typical usage is to pass extra args to evaluation env creator
        # and to disable exploration by computing deterministic actions
        "evaluation_config": {"explore": False},
    }
