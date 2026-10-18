"""Command-line entry point: ``greenqueue run eNN`` and ``greenqueue reproduce vN``."""

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
