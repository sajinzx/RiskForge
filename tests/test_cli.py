from typer.testing import CliRunner

import greenqueue
from greenqueue.cli import EXPERIMENTS, VERSIONS, app


def test_version_command() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == greenqueue.__version__


def test_unknown_experiment_and_version_are_rejected() -> None:
    assert CliRunner().invoke(app, ["run", "e99"]).exit_code != 0
    assert CliRunner().invoke(app, ["reproduce", "v9"]).exit_code != 0


def test_every_version_step_is_a_registered_experiment() -> None:
    assert all(name in EXPERIMENTS for names in VERSIONS.values() for name in names)
    assert VERSIONS["v1"] == ["e1", "e4"]
