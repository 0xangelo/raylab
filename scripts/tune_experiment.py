"""CLI for launching Tune experiments."""
import logging

import ray
from ray import tune
import click

import raylab
from raylab.logger import DEFAULT_LOGGERS as CUSTOM_LOGGERS
from raylab.utils.dynamic_import import import_module_from_path


@click.command()
@click.argument("run", type=str)
@click.option("--name", "-n", default=None, help="Name of experiment")
@click.option(
    "--local-dir",
    "-l",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
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
    "--stop",
    "-s",
    type=(str, int),
    multiple=True,
    help="The stopping criteria. "
    "The keys may be any field in the return result of 'train()', "
    "whichever is reached first. Defaults to empty dict.",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=False),
    default=None,
    help="Algorithm-specific configuration for Tune variant generation "
    "(e.g. env, hyperparams). Defaults to empty dict. "
    "Custom search algorithms may ignore this. "
    "Expects a path to a python script containing a `get_config` function. ",
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
    default="INFO",
    show_default=True,
    help="Logging level for the trial executor process. This is independent from each "
    "trainer's logging level.",
)
def main(**args):  # pylint: disable=missing-docstring
    if args["config"] is None:
        config = {}
    else:
        module = import_module_from_path(args["config"])
        config = module.get_config()

    raylab.register_all_agents()
    raylab.register_all_environments()

    ray.init(object_store_memory=args["object_store_memory"])
    logging.getLogger("ray.tune").setLevel(args["tune_log_level"])
    tune.run(
        args["run"],
        name=args["name"],
        local_dir=args["local_dir"],
        num_samples=args["num_samples"],
        stop={k: v for k, v in args["stop"]},
        config=config,
        checkpoint_freq=args["checkpoint_freq"],
        checkpoint_at_end=args["checkpoint_at_end"],
        loggers=CUSTOM_LOGGERS if args["custom_loggers"] else None,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
