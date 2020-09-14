# pylint:disable=missing-module-docstring
import click


@click.group("info")
def info_cli():
    """View information about an agent's config parameters."""


@info_cli.command("list")
@click.argument("agent")
@click.option("--key", type=str, default=None, help="Specific config to show info for.")
@click.option(
    "--separator",
    "-d",
    type=str,
    default="/",
    show_default=True,
    help="Separator for nested config keys.",
)
@click.option(
    "--rllib/--no-rllib",
    default=False,
    show_default=True,
    help="Whether to display RLlib's common config parameters and defaults. "
    "Warning: lots of parameters!",
)
@click.pass_context
def list_(ctx, agent, key, separator, rllib):
    """Retrieve and echo a help text for the given agent's config."""
    from raylab.options import UnknownOptionError
    from raylab.agents.registry import AGENTS

    cls = AGENTS[agent]()

    try:
        msg = cls.options.help(key, separator, with_rllib=rllib)
    except UnknownOptionError as err:
        click.echo(err)
        click.echo(f"Exception raised when querying agent {agent} options for help.")
        ctx.exit()

    click.echo(msg)
