"""CLI for launching Tune experiments."""
import click

from .utils import initialize_raylab, tune_options


@click.command()
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
@click.option(
    "--object-store-memory",
    type=int,
    default=int(2e9),
    show_default=True,
    help="The amount of memory (in bytes) to start the object store with. "
    "By default, this is capped at 20GB but can be set higher.",
)
@tune_options
@initialize_raylab
def experiment(run_or_experiment, config, object_store_memory, tune_kwargs):
    """Launch a Tune experiment from a config file."""
    import ray
    from ray import tune

    from raylab.utils.dynamic_import import import_module_from_path

    if config:
        config = import_module_from_path(config).get_config()

    ray.init(object_store_memory=object_store_memory)

    tune.run(run_or_experiment, config=config, **tune_kwargs)
