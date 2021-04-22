"""CLI utilities for RayLab."""
from __future__ import annotations

import click

from .best_checkpoint import find_best
from .evaluate_checkpoint import rollout
from .experiment import experiment
from .info import info_cli


@click.group()
def raylab():
    """RayLab: Reinforcement learning algorithms in RLlib."""


@raylab.command()
@click.argument(
    "paths",
    nargs=-1,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=False),
)
def dashboard(paths: tuple[str, ...]):
    """Launch the experiment dashboard to monitor training progress."""
    import subprocess

    from . import experiment_dashboard

    subprocess.run(
        ["streamlit", "run", experiment_dashboard.__file__] + list(paths), check=True
    )


@raylab.command()
@click.argument(
    "path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
)
def episodes(path: str):
    """Launch the episode dashboard to monitor state and action distributions."""
    import subprocess

    from . import episode_dashboard

    subprocess.run(["streamlit", "run", episode_dashboard.__file__, path], check=True)


@raylab.command()
@click.argument("agent_id", type=str)
@click.argument(
    "checkpoint",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
)
def test_module(agent_id, checkpoint):
    """Launch dashboard to test generative models from a checkpoint."""
    import subprocess

    from . import module_dashboard

    subprocess.run(
        ["streamlit", "run", module_dashboard.__file__, agent_id, checkpoint],
        check=True,
    )


raylab.add_command(experiment)
raylab.add_command(find_best)
raylab.add_command(rollout)
raylab.add_command(info_cli)
