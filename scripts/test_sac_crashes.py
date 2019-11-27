"""Dummy."""
import os
import os.path as osp

import click
import ray
from ray import tune

import raylab
from raylab.logger import DEFAULT_LOGGERS as CUSTOM_LOGGERS


@click.command()
@click.option(
    "--local-dir",
    "-l",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, resolve_path=True),
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
def main(**args):
    """Dummy experiments on Navigation."""
    if not osp.exists(args["local_dir"]) and click.confirm(
        "Provided `local_dir` does not exist. Create it?"
    ):
        os.makedirs(args["local_dir"])
        click.echo("Created directory {}".format(args["local_dir"]))

    raylab.register_all_agents()
    raylab.register_all_environments()
    ray.init(object_store_memory=args["object_store_memory"])

    tune.run(
        "SoftAC",
        name="SoftAC-Navigation-TestTrials",
        local_dir=args["local_dir"],
        num_samples=5,
        stop={"timesteps_total": 40},
        config={
            "env": "Navigation",
            "env_config": {"init_dist": False},
            # === Evaluation ===
            "evaluation_interval": 1,
            "evaluation_num_episodes": 5,
            # === RolloutWorker ===
            "sample_batch_size": 1,
            "batch_mode": "complete_episodes",
            # === Trainer ===
            "train_batch_size": 32,
            "timesteps_per_iteration": 20,
        },
        checkpoint_freq=args["checkpoint_freq"],
        checkpoint_at_end=args["checkpoint_at_end"],
        loggers=CUSTOM_LOGGERS,
    )


if __name__ == "__main__":
    main()
