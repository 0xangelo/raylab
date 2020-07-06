# pylint:disable=missing-module-docstring
import click


class UnknownConfigKeyError(Exception):
    """Exception raised for querying an unknown config key for help.

    Args:
        agent: Agent ID as a string
        key: Name of the config key (possibly nested) that is not in the agent's
            default config.
    """

    def __init__(self, key: str, agent: str):
        super().__init__(f"Key {key} is absent in {agent}'s config.")


@click.command()
@click.argument("agent")
@click.argument("config")
@click.option(
    "--separator",
    "-d",
    type=str,
    default="/",
    show_default=True,
    help="Separator for nested config keys.",
)
@click.pass_context
def info(ctx, agent, config, separator):
    """Retrieve and echo the help text for the given agent's config key."""
    from raylab.agents.registry import AGENTS

    cls = AGENTS[agent]()
    key_seq = config.split(separator)
    config_info = cls._config_info  # pylint:disable=protected-access

    try:
        config_info = find_config_info(key_seq, config_info, separator, agent)
    except UnknownConfigKeyError as err:
        click.echo(err)
        ctx.exit()

    click.echo(config_info)


def find_config_info(key_seq, config_info, separator, agent):
    """Find help for parameter in info dict.

    Args:
        key_seq: Hierarchy of nested parameter keys leading to the desired key
        config_info: The config info dictionary
        separator: Text token separating nested info keys
        agent: Agent's ID string

    Returns:
        The parameter's help text.

    Raises:
        UnknownConfigKeyError: If the search fails at any point in the key
            sequence
    """
    for idx, key in enumerate(key_seq):
        if key not in config_info:
            key = separator.join(key_seq[: idx + 1])
            raise UnknownConfigKeyError(key, agent)
        config_info = config_info[key]

    if isinstance(config_info, dict):
        config_info = config_info["__help__"]

    return config_info
