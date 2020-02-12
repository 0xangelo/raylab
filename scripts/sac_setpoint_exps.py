# pylint:disable=missing-docstring
import os
import os.path as osp
import logging

import click
import seaborn as sns
import matplotlib.pyplot as plt
import ray
from ray import tune

import raylab
from raylab.cli.viskit import core
from raylab.cli.viskit.plot import plot_figures
from raylab.logger import DEFAULT_LOGGERS as CUSTOM_LOGGERS


def get_config():
    return {
        # === Environment ===
        "env": "IndustrialBenchmark",
        "env_config": {
            "setpoint": tune.grid_search([(i + 1) * 10 for i in range(10)]),
            "reward_type": "classic",
            "action_type": "continuous",
            "observation": "markovian",
            "max_episode_steps": 1000,
            "time_aware": True,
        },
        # === Replay Buffer ===
        "buffer_size": int(1e5),
        # === Optimization ===
        # PyTorch optimizer to use for policy
        "policy_optimizer": {"name": "Adam", "options": {"lr": 3e-4}},
        # PyTorch optimizer to use for critic
        "critic_optimizer": {"name": "Adam", "options": {"lr": 3e-4}},
        # PyTorch optimizer to use for entropy coefficient
        "alpha_optimizer": {"name": "Adam", "options": {"lr": 3e-4}},
        # === Network ===
        # Size and activation of the fully connected networks computing the logits
        # for the policy and action-value function. No layers means the component is
        # linear in states and/or actions.
        "module": {
            "policy": {
                "units": (256, 256),
                "activation": "ReLU",
                "initializer_options": {"name": "xavier_uniform"},
                "input_dependent_scale": True,
            },
            "critic": {
                "units": (256, 256),
                "activation": "ReLU",
                "initializer_options": {"name": "xavier_uniform"},
                "delay_action": True,
            },
        },
        # === Exploration ===
        "pure_exploration_steps": 2000,
        # === Trainer ===
        "train_batch_size": 256,
        "timesteps_per_iteration": 1000,
        # === Evaluation ===
        # Evaluate with every `evaluation_interval` training iterations.
        # The evaluation stats will be reported under the "evaluation" metric key.
        "evaluation_interval": 1,
        # Number of episodes to run per evaluation period.
        "evaluation_num_episodes": 5,
        # === Debugging ===
        # Set the ray.rllib.* log level for the agent process and its workers.
        # Should be one of DEBUG, INFO, WARN, or ERROR. The DEBUG level will also
        # periodically print out summaries of relevant internal dataflow (this is
        # also printed out once at startup at the INFO level).
        "log_level": "WARN",
    }


@click.group()
def cli():
    pass


@cli.command()
@click.option("--name", default=None, help="Name of experiment")
@click.option(
    "--local-dir",
    "-l",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, resolve_path=True),
    default="data/",
    show_default=True,
    help="",
)
@click.option(
    "--num-samples",
    "-n",
    type=int,
    default=1,
    show_default=True,
    help="Number of times to sample from the hyperparameter space. "
    "Defaults to 1. If `grid_search` is provided as an argument, "
    "the grid will be repeated `num_samples` of times.",
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
    "--custom-loggers/--no-custom-loggers",
    default=True,
    help="Use custom loggers from raylab.logger.",
)
@click.option(
    "--tune-log-level",
    type=str,
    default="WARN",
    show_default=True,
    help="Logging level for the trial executor process. This is independent from each "
    "trainer's logging level.",
)
@click.pass_context
def experiment(ctx, **args):
    if not osp.exists(args["local_dir"]) and click.confirm(
        "Provided `local_dir` does not exist. Create it?"
    ):
        os.makedirs(args["local_dir"])
        click.echo("Created directory {}".format(args["local_dir"]))

    exp_dir = osp.join(args["local_dir"], args["name"])
    if osp.exists(exp_dir) and not click.confirm(
        f"Experiment directory {exp_dir} already exists. Proceed anyway?"
    ):
        ctx.exit()

    raylab.register_all_agents()
    raylab.register_all_environments()
    ray.init(object_store_memory=args["object_store_memory"])
    logging.getLogger("ray.tune").setLevel(args["tune_log_level"])
    tune.run(
        "SoftAC",
        name=args["name"],
        local_dir=args["local_dir"],
        num_samples=args["num_samples"],
        stop={"timesteps_total": int(1e4)},
        config=get_config(),
        checkpoint_freq=args["checkpoint_freq"],
        checkpoint_at_end=args["checkpoint_at_end"],
        loggers=CUSTOM_LOGGERS if args["custom_loggers"] else None,
    )


@cli.command()
@click.option(
    "--path",
    "-p",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, resolve_path=True),
)
def plot(path):
    exps_data = core.load_exps_data(path)
    for exp_data in exps_data:
        exp_data.flat_params["agent"] = "SAC"
    core.insert_params_dataframe(exps_data, "agent", "env_config/setpoint")

    selectors, titles = core.filter_and_split_experiments(
        exps_data, split="env_config/setpoint"
    )
    instructions = core.lineplot_instructions(
        selectors,
        titles,
        x="timesteps_total",
        y="evaluation/episode_reward_mean",
        legend="full",
    )
    instructions = sorted(instructions, key=lambda x: int(x["title"]))
    with sns.plotting_context("paper"), sns.axes_style("darkgrid"):
        plot_figures(instructions)
        plt.show()


if __name__ == "__main__":
    cli()  # pylint:disable=no-value-for-parameter
