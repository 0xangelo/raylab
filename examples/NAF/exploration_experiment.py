"""Tune experiment configuration to compare exploration options in NAF.

This can be run from the command line by executing
`raylab experiment NAF --config examples/naf_exploration_experiment.py -s timesteps_total 100000`
"""
from ray import tune


def get_config():
    return {
        # === Environment ===
        "env": "TimeLimitedEnv",
        "env_config": {
            "env_id": "CartPoleSwingUp",
            "max_episode_steps": 500,
            "time_aware": False,
        },
        # === SQUASHING EXPLORATION PROBLEM ===
        # Maximum l1 norm of the policy's output vector before the squashing function
        "beta": 1.2,
        # === Twin Delayed DDPG (TD3) tricks ===
        # Clipped Double Q-Learning
        "clipped_double_q": False,
        # === Replay buffer ===
        # Size of the replay buffer. Note that if async_updates is set, then
        # each worker will have a replay buffer of this size.
        "buffer_size": int(1e5),
        # === Network ===
        # Size and activation of the fully connected network computing the logits
        # for the normalized advantage function. No layers means the Q function is
        # linear in states and actions.
        "module": {
            "units": (128, 128),
            "activation": "ReLU",
            "initializer_options": {"name": "xavier_uniform"},
        },
        # === Optimization ===
        # Name of Pytorch optimizer class
        "torch_optimizer": {"name": "Adam", "options": {"lr": 3e-4}},
        # Interpolation factor in polyak averaging for target networks.
        "polyak": 0.995,
        # === Rollout Worker ===
        "num_workers": 0,
        "sample_batch_size": 1,
        "batch_mode": "complete_episodes",
        # === Trainer ===
        "train_batch_size": 128,
        "timesteps_per_iteration": 1000,
        # === Exploration ===
        # Which type of exploration to use. Possible types include
        # None: use the greedy policy to act
        # parameter_noise: use parameter space noise
        # diag_gaussian: use i.i.d gaussian action space noise independently for each
        #     action dimension
        # full_gaussian: use gaussian action space noise where the precision matrix is
        #     given by the advantage function P matrix
        "exploration": tune.grid_search(["diag_gaussian", "parameter_noise"]),
        # Gaussian stddev for diagonal gaussian action space noise
        "diag_gaussian_stddev": 0.3,
        # Until this many timesteps have elapsed, the agent's policy will be
        # ignored & it will instead take uniform random actions. Can be used in
        # conjunction with learning_starts (which controls when the first
        # optimization step happens) to decrease dependence of exploration &
        # optimization on initial policy parameters. Note that this will be
        # disabled when the action noise scale is set to 0 (e.g during evaluation).
        "pure_exploration_steps": 5000,
        # Options for parameter noise exploration
        "param_noise_spec": {
            "initial_stddev": 0.1,
            "desired_action_stddev": 0.3,
            "adaptation_coeff": 1.01,
        },
        # === Evaluation ===
        # Evaluate with every `evaluation_interval` training iterations.
        # The evaluation stats will be reported under the "evaluation" metric key.
        "evaluation_interval": 5,
        # === Debugging ===
        # Set the ray.rllib.* log level for the agent process and its workers.
        # Should be one of DEBUG, INFO, WARN, or ERROR. The DEBUG level will also
        # periodically print out summaries of relevant internal dataflow (this is
        # also printed out once at startup at the INFO level).
        "log_level": "WARN",
    }
