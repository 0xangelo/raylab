"""CLI utilities for RayLab."""
import click

from raylab.cli.tune_experiment import experiment


@click.group()
def cli():
    """RayLab: Reinforcement learning algorithms in RLlib."""


cli.add_command(experiment)
