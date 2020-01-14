"""CLI utilities for RayLab."""
import click

import raylab
from raylab.cli.tune_experiment import experiment
from raylab.cli.best_checkpoint import find_best
from raylab.cli.evaluate_checkpoint import rollout
from raylab.cli.viskit import plot, plot_export


@click.group()
def cli():
    """RayLab: Reinforcement learning algorithms in RLlib."""
    raylab.register_all_agents()
    raylab.register_all_environments()


cli.add_command(experiment)
cli.add_command(find_best)
cli.add_command(rollout)
cli.add_command(plot)
cli.add_command(plot_export)
