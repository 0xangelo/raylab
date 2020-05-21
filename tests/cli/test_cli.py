# pylint: disable=missing-docstring,redefined-outer-name,protected-access
from click.testing import CliRunner

from raylab import cli


def test_command_line_interface():
    runner = CliRunner()
    result = runner.invoke(cli.raylab)
    assert result.exit_code == 0
    assert "raylab" in result.output
    help_result = runner.invoke(cli.raylab, ["--help"])
    assert help_result.exit_code == 0
    assert "--help  Show this message and exit." in help_result.output
