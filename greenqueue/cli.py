"""Command-line entry point: ``greenqueue run eNN`` and ``greenqueue reproduce vN``."""

from pathlib import Path

import typer

import greenqueue

app = typer.Typer(help="GreenQueue: uncertainty-aware carbon scheduling.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """GreenQueue command group."""


@app.command()
def version() -> None:
    """Print the installed GreenQueue version."""
    typer.echo(greenqueue.__version__)


@app.command()
def run(
    experiment: str = typer.Argument(..., help="Experiment id, e.g. e1 or e4"),
    smoke: bool = typer.Option(
        False, help="Short version to check the pipeline (e3: synthetic data)"
    ),
    trials: int | None = typer.Option(None, help="Override Optuna trials (e1)"),
    root: Path = typer.Option(Path("."), help="Project root"),
) -> None:
    """Run one experiment and write results/<experiment>/."""
    name = experiment.lower()
    if name == "e1":
        from greenqueue.experiments.e1 import run_e1

        summary = run_e1(root.resolve(), smoke=smoke, trials=trials)
    elif name == "e3":
        from greenqueue.experiments.e3 import run_e3

        summary = run_e3(root.resolve(), synthetic=smoke)
    elif name == "e4":
        from greenqueue.experiments.e4 import run_e4

        summary = run_e4(root.resolve(), smoke=smoke)
    else:
        raise typer.BadParameter(f"unknown experiment {experiment!r}")
    typer.echo(f"{name} done, config hash {summary['config_hash']}")


if __name__ == "__main__":
    app()
