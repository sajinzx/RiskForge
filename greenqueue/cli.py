"""Command-line entry point: ``greenqueue run eNN`` and ``greenqueue reproduce vN``."""

import importlib
from pathlib import Path
from typing import Any

import typer

import greenqueue

app = typer.Typer(help="GreenQueue: uncertainty-aware carbon scheduling.", no_args_is_help=True)

# experiment id -> (module, function)
EXPERIMENTS = {
    "e1": ("greenqueue.experiments.e1", "run_e1"),
    "e2": ("greenqueue.experiments.e2", "run_e2"),
    "e3": ("greenqueue.experiments.e3", "run_e3"),
    "e4": ("greenqueue.experiments.e4", "run_e4"),
    "e5": ("greenqueue.experiments.e5", "run_e5"),
    "e6": ("greenqueue.experiments.e6", "run_e6"),
    "e7": ("greenqueue.experiments.e7", "run_e7"),
    "ew": ("greenqueue.experiments.ew", "run_ew"),
}
# experiments needed to rebuild each version's figures, in dependency order
VERSIONS = {
    "v1": ["e1", "e4"],
    "v2": ["e1", "e2", "ew", "e3", "e4", "e5", "e6", "e7"],
}


@app.callback()
def main() -> None:
    """GreenQueue command group."""


@app.command()
def version() -> None:
    """Print the installed GreenQueue version."""
    typer.echo(greenqueue.__version__)


def _run(name: str, root: Path, smoke: bool, trials: int | None = None) -> dict[str, Any]:
    if name not in EXPERIMENTS:
        raise typer.BadParameter(f"unknown experiment {name!r}")
    module, func = EXPERIMENTS[name]
    fn = getattr(importlib.import_module(module), func)
    if name == "e1":
        return fn(root, smoke=smoke, trials=trials)  # type: ignore[no-any-return]
    if name == "e3":
        return fn(root, synthetic=smoke)  # type: ignore[no-any-return]
    return fn(root, smoke=smoke)  # type: ignore[no-any-return]


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
    summary = _run(experiment.lower(), root.resolve(), smoke, trials)
    typer.echo(f"{experiment} done, config hash {summary['config_hash']}")


@app.command()
def reproduce(
    target: str = typer.Argument(..., help="v1 or v2"),
    smoke: bool = typer.Option(False, help="Short versions of every experiment"),
    root: Path = typer.Option(Path("."), help="Project root"),
) -> None:
    """Rebuild every figure and table of a version, in dependency order.

    Needs the raw data in place (see data/MANIFEST.md): run the two download scripts first.
    v2 also needs GPU measurements for E3; without them E3 stops with an error.
    """
    if target not in VERSIONS:
        raise typer.BadParameter("target must be v1 or v2")
    for name in VERSIONS[target]:
        typer.echo(f"== {name}")
        summary = _run(name, root.resolve(), smoke)
        typer.echo(f"   config hash {summary['config_hash']}")


if __name__ == "__main__":
    app()
