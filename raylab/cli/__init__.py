"""CLI utilities for RayLab."""
import click

from .tune_experiment import experiment
from .best_checkpoint import find_best
from .evaluate_checkpoint import rollout
from .viskit import plot, plot_export


@click.group()
def cli():
    """RayLab: Reinforcement learning algorithms in RLlib."""


cli.add_command(experiment)
cli.add_command(find_best)
cli.add_command(rollout)
cli.add_command(plot)
cli.add_command(plot_export)
