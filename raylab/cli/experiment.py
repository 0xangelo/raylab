"""CLI for launching Tune experiments."""
import click

from .utils import tune_experiment


@tune_experiment
@click.argument("run_or_experiment", type=str)
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
def experiment(run_or_experiment, config):
    """Launch a Tune experiment from a config file."""
    from raylab.utils.dynamic_import import import_module_from_path

    if config:
        config = import_module_from_path(config).get_config()

    return run_or_experiment, config, {}
