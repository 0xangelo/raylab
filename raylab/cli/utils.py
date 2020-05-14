# pylint:disable=missing-module-docstring
import functools
import os
import os.path as osp
import logging

import click


def initialize_raylab(func):
    """Wrap cli to register raylab's algorithms and environments."""

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        import raylab

        raylab.register_all_agents()
        raylab.register_all_environments()

        return func(*args, **kwargs)

    return wrapped


def tune_options(func):
    """Wrap cli to add and parse arguments for `tune.run`."""

    @click.option("--name", default=None, help="Name of experiment")
    @click.option(
        "--local-dir",
        "-l",
        type=click.Path(
            exists=False, file_okay=False, dir_okay=True, resolve_path=True
        ),
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
        "--custom-loggers/--no-custom-loggers",
        default=True,
        help="Use custom loggers from raylab.logger.",
    )
    @click.option(
        "--tune-log-level",
        type=str,
        default="WARN",
        show_default=True,
        help="Logging level for the trial executor process. This is independent from "
        "each trainer's logging level.",
    )
    @click.option(
        "--restore",
        type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
        default=None,
        show_default=True,
        help="Path to checkpoint. Only makes sense to set if running 1 trial.",
    )
    @click.pass_context
    @functools.wraps(func)
    def wrapped(
        ctx,
        name,
        local_dir,
        num_samples,
        stop,
        checkpoint_freq,
        checkpoint_at_end,
        custom_loggers,
        tune_log_level,
        restore,
        *args,
        **kwargs,
    ):
        # pylint:disable=too-many-arguments
        from raylab.logger import DEFAULT_LOGGERS as CUSTOM_LOGGERS

        tune_kwargs = dict(
            name=name,
            local_dir=local_dir,
            num_samples=num_samples,
            checkpoint_freq=checkpoint_freq,
            checkpoint_at_end=checkpoint_at_end,
            restore=restore,
        )
        tune_kwargs["loggers"] = CUSTOM_LOGGERS if custom_loggers else None
        tune_kwargs["stop"] = dict(stop)

        setup_tune_logger(tune_log_level)
        create_if_necessary(local_dir)
        delete_if_necessary(ctx, osp.join(local_dir, name))

        return func(*args, tune_kwargs=tune_kwargs, **kwargs)

    return wrapped


def setup_tune_logger(tune_log_level):
    """Set Tune's log level."""
    logging.getLogger("ray.tune").setLevel(tune_log_level)


def create_if_necessary(directory):
    """Ask user to allow directory creation if necessary."""
    msg = f"Directory {directory} does not exist. Create it?"
    if not osp.exists(directory) and click.confirm(msg):
        os.makedirs(directory)
        click.echo(f"Created directory {directory}")


def delete_if_necessary(ctx, exp_dir):
    """Ask user to allow directory deletion if necessary."""
    if osp.exists(exp_dir):
        msg = f"Experiment directory {exp_dir} already exists. Remove and proceed?"
        if not click.confirm(msg):
            ctx.exit()

        import shutil

        shutil.rmtree(exp_dir)
        click.echo(f"Removed directory {exp_dir}")
