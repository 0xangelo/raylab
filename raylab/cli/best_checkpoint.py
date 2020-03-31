"""CLI for finding the best checkpoint of an experiment."""
import click

from .utils import initialize_raylab


def get_last_checkpoint_path(logdir):
    """Retrieve the path of the last checkpoint given a Trial logdir."""
    import os.path as osp
    from glob import glob

    last_checkpoint_basedir = sorted(
        glob(osp.join(logdir, "checkpoint_*")), key=lambda p: p.split("_")[-1]
    )[-1]
    last_checkpoint_path = osp.join(
        last_checkpoint_basedir, osp.basename(last_checkpoint_basedir).replace("_", "-")
    )
    return last_checkpoint_path


@click.command()
@click.argument(
    "logdir",
    nargs=1,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
)
@click.option(
    "--metric",
    default="episode_reward_mean",
    show_default=True,
    help="Key for trial info to order on.",
)
@click.option(
    "--mode",
    type=click.Choice("max min".split()),
    default="max",
    show_default=True,
    help="Criterion to order trials by.",
)
@initialize_raylab
def find_best(logdir, metric, mode):
    """Find the best experiment checkpoint as measured by a metric."""
    import logging

    from ray.tune.analysis import Analysis

    logging.getLogger("ray.tune").setLevel("ERROR")

    analysis = Analysis(logdir)
    best_logdir = analysis.get_best_logdir(metric, mode=mode)
    last_checkpoint_path = get_last_checkpoint_path(best_logdir)
    click.echo(last_checkpoint_path)
