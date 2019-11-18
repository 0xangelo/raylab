"""Script for launching Tune experiments for MAPO."""
import logging

import click
import ray
from ray import tune

import raylab
from raylab.logger import DEFAULT_LOGGERS as CUSTOM_LOGGERS
from plot_icaps_grid import plot_grid


STOP_COND = {"timesteps_total": int(2e4)}

DECELERATION_ZONES = tune.grid_search(
    [
        {"center": [[0.0, 0.0]], "decay": [2.0]},
        {"center": [[-2.0, 0.0], [2.0, 0.0]], "decay": [2.0, 2.0]},
        {
            "center": [[-2.0, -2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, 2.0]],
            "decay": [2.0, 2.0, 2.0, 2.0],
        },
    ]
)

ACTOR_CRITIC_CONFIG = {
    "policy": {
        "units": (64,),
        "activation": "ReLU",
        "initializer_options": {"name": "xavier_uniform"},
    },
    "critic": {
        "units": (64,),
        "activation": "ReLU",
        "initializer_options": {"name": "xavier_uniform"},
        "delay_action": True,
    },
}

Q_LEARNING_CONFIG = {
    # === SQUASHING EXPLORATION PROBLEM ===
    # Maximum l1 norm of the policy's output vector before the squashing function
    "beta": 1.2,
    # === Twin Delayed DDPG (TD3) tricks ===
    # Clipped Double Q-Learning: use the minimun of two target Q functions
    # as the next action-value in the target for fitted Q iteration
    "clipped_double_q": True,
    # Add gaussian noise to the action when calculating the Deterministic
    # Policy Gradient
    "target_policy_smoothing": True,
    # Additive Gaussian i.i.d. noise to add to actions inputs to target Q function
    "target_gaussian_sigma": 0.3,
    # Interpolation factor in polyak averaging for target networks.
    "polyak": 0.995,
}

REPLAY_CONFIG = {
    # === Replay Buffer ===
    "buffer_size": int(1e4)
}

EXPLORATION_CONFIG = {
    # === Exploration ===
    # Which type of exploration to use. Possible types include
    # None: use the greedy policy to act
    # parameter_noise: use parameter space noise
    # gaussian: use i.i.d gaussian action space noise independently for each
    #     action dimension
    "exploration": "gaussian",
    # Additive Gaussian i.i.d. noise to add to actions before squashing
    "exploration_gaussian_sigma": 0.3,
    # Until this many timesteps have elapsed, the agent's policy will be
    # ignored & it will instead take uniform random actions. Can be used in
    # conjunction with learning_starts (which controls when the first
    # optimization step happens) to decrease dependence of exploration &
    # optimization on initial policy parameters. Note that this will be
    # disabled when the action noise scale is set to 0 (e.g during evaluation).
    "pure_exploration_steps": 200,
}

EVALUATION_CONFIG = {
    # === Evaluation ===
    "evaluation_interval": 5,
    "evaluation_num_episodes": 5,
}

TRAINER_CONFIG = {
    # === RolloutWorker ===
    "sample_batch_size": 1,
    "batch_mode": "complete_episodes",
    # === Trainer ===
    "train_batch_size": 32,
    "timesteps_per_iteration": 200,
    # === Debugging ===
    # Set the ray.rllib.* log level for the agent process and its workers.
    # Should be one of DEBUG, INFO, WARN, or ERROR. The DEBUG level will also
    # periodically print out summaries of relevant internal dataflow (this is
    # also printed out once at startup at the INFO level).
    "log_level": "WARN",
}


MAPO_CONFIG = {
    # === MAPO model training ===
    # Type of model-training to use. Possible types include
    # decision_aware: policy gradient-aware model learning
    # mle: maximum likelihood estimation
    "model_loss": tune.grid_search(["decision_aware", "mle"]),
    # Gradient estimator for model-aware dpg. Possible types include:
    # score_function, pathwise_derivative
    "grad_estimator": tune.grid_search(["score_function", "pathwise_derivative"]),
    # Type of the used p-norm of the distance between gradients.
    # Can be float('inf') for infinity norm.
    "norm_type": 2,
    # Number of next states to sample from the model when calculating the
    # model-aware deterministic policy gradient
    "num_model_samples": 4,
    # === Optimization ===
    # PyTorch optimizer to use for policy
    "policy_optimizer": {"name": "Adam", "options": {"lr": 3e-4}},
    # PyTorch optimizer to use for critic
    "critic_optimizer": {"name": "Adam", "options": {"lr": 3e-4}},
    # PyTorch optimizer to use for model
    "model_optimizer": {"name": "Adam", "options": {"lr": 3e-4}},
}

SOP_CONFIG = {
    # === Optimization ===
    # PyTorch optimizer to use for policy
    "policy_optimizer": {"name": "Adam", "options": {"lr": 3e-4}},
    # PyTorch optimizer to use for critic
    "critic_optimizer": {"name": "Adam", "options": {"lr": 3e-4}},
    # === Network ===
    # Size and activation of the fully connected networks computing the logits
    # for the policy and action-value function. No layers means the component is
    # linear in states and/or actions.
    "module": ACTOR_CRITIC_CONFIG,
}


@click.command()
@click.option(
    "--local-dir",
    "-l",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    default="data/",
    show_default=True,
    help="",
)
@click.option(
    "--checkpoint-freq",
    type=int,
    default=0,
    show_default=True,
    help="How many training iterations between checkpoints. "
    "A value of 0 disables checkpointing.",
)
@click.option(
    "--checkpoint-at-end",
    type=bool,
    default=True,
    show_default=True,
    help="Whether to checkpoint at the end of the experiment regardless of "
    "the checkpoint_freq.",
)
@click.option(
    "--object-store-memory",
    type=int,
    default=int(2e9),
    show_default=True,
    help="The amount of memory (in bytes) to start the object store with. "
    "By default, this is capped at 20GB but can be set higher.",
)
@click.option(
    "--tune-log-level",
    type=str,
    default="WARN",
    show_default=True,
    help="Logging level for the trial executor process. This is independent from each "
    "trainer's logging level.",
)
def experiment(**args):
    """Launch a Tune experiment for MAPO on Navigation."""
    raylab.register_all_agents()
    raylab.register_all_environments()
    ray.init(object_store_memory=args["object_store_memory"])
    logging.getLogger("ray.tune").setLevel(args["tune_log_level"])

    base_config = {
        **Q_LEARNING_CONFIG,
        **REPLAY_CONFIG,
        **EXPLORATION_CONFIG,
        **EVALUATION_CONFIG,
        **TRAINER_CONFIG,
        "seed": tune.grid_search(list(range(10))),
    }

    # === Random Walk Experiments ===
    env_config = {
        "env": "Navigation",
        "env_config": {
            "deceleration_zones": DECELERATION_ZONES,
            "random_walks": {"num_walks": 8, "loc": 10.0, "scale": 2.0},
        },
    }
    mapo_config = {
        **MAPO_CONFIG,
        "module": {
            **ACTOR_CRITIC_CONFIG,
            "model": {
                "units": (3,),
                "activation": "ReLU",
                "initializer_options": {"name": "xavier_uniform"},
                "delay_action": True,
                "input_dependent_scale": False,
            },
        },
    }

    tune.run(
        "MAPO",
        name="MAPO-Navigation-Walks",
        local_dir=args["local_dir"],
        stop=STOP_COND,
        config={**base_config, **env_config, **mapo_config},
        checkpoint_freq=args["checkpoint_freq"],
        checkpoint_at_end=args["checkpoint_at_end"],
        loggers=CUSTOM_LOGGERS,
    )
    tune.run(
        "SOP",
        name="SOP-Navigation-Walks",
        local_dir=args["local_dir"],
        stop=STOP_COND,
        config={**base_config, **env_config, **SOP_CONFIG},
        checkpoint_freq=args["checkpoint_freq"],
        checkpoint_at_end=args["checkpoint_at_end"],
        loggers=CUSTOM_LOGGERS,
    )

    # === Non-Diag Covariance Experiments ===
    env_config = {
        "env": "Navigation",
        "env_config": {
            "deceleration_zones": DECELERATION_ZONES,
            "noise": {"loc": [0.0, 0.0], "scale_tril": [[0.3, 0.0], [0.2, 0.3]]},
        },
    }
    mapo_config = {
        **MAPO_CONFIG,
        "module": {
            **ACTOR_CRITIC_CONFIG,
            "model": {
                "units": (64,),
                "activation": "ReLU",
                "initializer_options": {"name": "xavier_uniform"},
                "delay_action": True,
                "input_dependent_scale": False,
            },
        },
    }

    tune.run(
        "MAPO",
        name="MAPO-Navigation-NonDiagCov",
        local_dir=args["local_dir"],
        stop=STOP_COND,
        config={**base_config, **env_config, **mapo_config},
        checkpoint_freq=args["checkpoint_freq"],
        checkpoint_at_end=args["checkpoint_at_end"],
        loggers=CUSTOM_LOGGERS,
    )
    tune.run(
        "SOP",
        name="SOP-Navigation-NonDiagCov",
        local_dir=args["local_dir"],
        stop=STOP_COND,
        config={**base_config, **env_config, **SOP_CONFIG},
        checkpoint_freq=args["checkpoint_freq"],
        checkpoint_at_end=args["checkpoint_at_end"],
        loggers=CUSTOM_LOGGERS,
    )

    # === Linear Model Experiments ===
    env_config = {
        # === Environment ===
        "env": "Navigation",
        "env_config": {"deceleration_zones": DECELERATION_ZONES},
    }
    mapo_config = {
        **MAPO_CONFIG,
        "module": {
            **ACTOR_CRITIC_CONFIG,
            "model": {
                "units": (),
                "activation": None,
                "initializer_options": {"name": "xavier_uniform"},
                "delay_action": True,
                "input_dependent_scale": False,
            },
        },
    }

    tune.run(
        "MAPO",
        name="MAPO-Navigation-LinearModel",
        local_dir=args["local_dir"],
        stop=STOP_COND,
        config={**base_config, **env_config, **mapo_config},
        checkpoint_freq=args["checkpoint_freq"],
        checkpoint_at_end=args["checkpoint_at_end"],
        loggers=CUSTOM_LOGGERS,
    )
    tune.run(
        "SOP",
        name="SOP-Navigation-LinearModel",
        local_dir=args["local_dir"],
        stop=STOP_COND,
        config={**base_config, **env_config, **SOP_CONFIG},
        checkpoint_freq=args["checkpoint_freq"],
        checkpoint_at_end=args["checkpoint_at_end"],
        loggers=CUSTOM_LOGGERS,
    )

    plot_grid(args["local_dir"])


if __name__ == "__main__":
    experiment()
