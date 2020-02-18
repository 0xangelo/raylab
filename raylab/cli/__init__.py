"""CLI utilities for RayLab."""
import click

from .tune_experiment import experiment
from .best_checkpoint import find_best
from .evaluate_checkpoint import rollout
from .viskit import plot, plot_export


@click.group()
def cli():
    """RayLab: Reinforcement learning algorithms in RLlib."""


@cli.command()
@click.argument(
    "paths",
    nargs=-1,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=False),
)
def dashboard(paths):
    """Launch the experiment dashboard to monitor training progress."""
    from streamlit.cli import _main_run
    from . import experiment_dashboard

    _main_run(experiment_dashboard.__file__, paths)


cli.add_command(experiment)
cli.add_command(find_best)
cli.add_command(rollout)
cli.add_command(plot)
cli.add_command(plot_export)
