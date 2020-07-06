# pylint:disable=missing-module-docstring
import textwrap
from typing import Dict
from typing import Union

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


MAX_REPR_LEN = 40


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
    help="Whether to display RLlib's common config parameters and defaults"
    "Warning: lots of parameters!",
)
@click.pass_context
def list_(ctx, agent, key, separator, rllib):
    """Retrieve and echo a help text for the given agent's config."""
    from raylab.agents.config import COMMON_INFO
    from raylab.agents.registry import AGENTS

    cls = AGENTS[agent]()

    if key is not None:
        try:
            msg = find_config_info(cls, key, separator)
        except UnknownConfigKeyError as err:
            click.echo(err)
            ctx.exit()

        click.echo(msg)
        ctx.exit()

    config = cls._default_config  # pylint:disable=protected-access
    info = cls._config_info  # pylint:disable=protected-access
    toplevel_keys = set(info.keys())
    if not rllib:
        toplevel_keys.difference_update(set(COMMON_INFO.keys()))

    for key_ in sorted(toplevel_keys):
        click.echo(parse_info(config, info, key_))


def find_config_info(cls: type, key: str, separator: str) -> str:
    """Find help for parameter in info dict.

    Args:
        cls: Agent's trainer class
        key: Hierarchy of nested parameter keys leading to the desired key
        separator: Text token separating nested info keys

    Returns:
        The parameter's help text.

    Raises:
        UnknownConfigKeyError: If the search fails at any point in the key
            sequence
    """
    key_seq = key.split(separator)
    config = cls._default_config  # pylint:disable=protected-access
    info = cls._config_info  # pylint:disable=protected-access

    def check_help(k, i, seq):
        if k not in i:
            k = separator.join(seq)
            # pylint:disable=protected-access
            raise UnknownConfigKeyError(key_, cls._name)

    for idx, key_ in enumerate(key_seq[:-1]):
        check_help(key_, info, key_seq[: idx + 1])
        config = config[key_]
        info = info[key_]
    key_ = key_seq[-1]
    check_help(key_, info, key_seq)

    return parse_info(config, info, key_)


def parse_info(config: dict, info: Dict[str, Union[str, dict]], key: str) -> str:
    """Returns the string form of the parameter info."""
    default = repr(config[key])
    if len(default) > MAX_REPR_LEN:
        default = repr(type(config[key]))

    help_ = info[key]
    if isinstance(help_, dict):
        help_ = help_["__help__"]

    msg = f"{key}: {default}\n"
    msg = msg + textwrap.indent(f"{help_}", prefix=" " * 4)
    return msg
