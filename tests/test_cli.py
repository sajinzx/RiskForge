from typer.testing import CliRunner

import greenqueue
from greenqueue.cli import app


def test_version_command() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == greenqueue.__version__
