"""CLI utilities for RayLab."""
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
def dashboard(paths):
    """Launch the experiment dashboard to monitor training progress."""
    from streamlit.cli import _main_run
    from . import experiment_dashboard

    _main_run(experiment_dashboard.__file__, paths)


@raylab.command()
@click.argument(
    "path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
)
def episodes(path):
    """Launch the episode dashboard to monitor state and action distributions."""
    from streamlit.cli import _main_run
    from . import episode_dashboard

    _main_run(episode_dashboard.__file__, [path])


@raylab.command()
@click.argument("agent_id", type=str)
@click.argument(
    "checkpoint",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
)
def test_module(agent_id, checkpoint):
    """Launch dashboard to test generative models from a checkpoint."""
    from streamlit.cli import _main_run
    from . import module_dashboard

    _main_run(module_dashboard.__file__, [agent_id, checkpoint])


raylab.add_command(experiment)
raylab.add_command(find_best)
raylab.add_command(rollout)
raylab.add_command(info_cli)
