"""Command-line interface — the root Typer app, and the sub-app per lane hung off it.

Every command that names a **Database**, an **Embedding** or a search ships from the module
that owns it, beside the result type it renders. What is left here is the two commands
belonging to no lane and the :meth:`typer.Typer.add_typer` calls that mount the rest.

Examples
--------
>>> from protein.cli import app
>>> [command.name for command in app.registered_commands]
['version', 'doctor']
"""

from __future__ import annotations

import json as _json

import typer

from protein import __version__ as _package_version
from protein import msa as _msa_cli
from protein import sifts as _sifts_cli
from protein.db import cli as _db_cli
from protein.embed import cli as _esm_cli
from protein.external import ToolNotFoundError as _ToolNotFoundError
from protein.external import doctor as _doctor
from protein.search import cli as _search_cli
from protein.structure import cli as _structure_cli

app = typer.Typer(help="Tools for handling protein sequences and structures.", no_args_is_help=True)


@app.command("version")
def version(
    json: bool = typer.Option(False, "--json", help="Emit JSON instead of plain text."),
) -> None:
    """Print the installed package version."""
    if json:
        typer.echo(_json.dumps({"version": _package_version}))
        return
    typer.echo(_package_version)


@app.command("doctor")
def doctor(
    json: bool = typer.Option(False, "--json", help="Emit JSON instead of plain text."),
) -> None:
    """Report availability and versions of the required native tools.

    Exits with code 1 if any one of them is missing from PATH; the message names the command
    that installs it.
    """
    try:
        versions = _doctor()
    except _ToolNotFoundError as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(code=1) from err

    if json:
        typer.echo(_json.dumps(versions))
        return
    for name, reported in versions.items():
        typer.echo(f"{name}: {reported}")


# --- the lane sub-apps -------------------------------------------------------
#
# One `add_typer` per lane, mounted here and nowhere else.
#
app.add_typer(_db_cli.app, name="db")
app.add_typer(_esm_cli.app, name="esm")
app.add_typer(_msa_cli.app, name="msa")
app.add_typer(_search_cli.app, name="search")
app.add_typer(_sifts_cli.app, name="sifts")
app.add_typer(_structure_cli.app, name="structure")
