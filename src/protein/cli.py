"""Command-line interface — the root Typer app, and the sub-app per lane hung off it.

Every command that names a **Database**, an **Embedding** or a search ships from the module
that owns it, beside the result type it renders. What is left here is the two commands
belonging to no lane — ``version`` and ``doctor`` — and the :meth:`typer.Typer.add_typer`
calls that mount the lanes.

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
from protein.external import ToolNotFoundError as _ToolNotFoundError
from protein.external import doctor as _doctor

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

    Exits with code 1 if either tool is missing from PATH; the message names the command
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
# One `add_typer` per lane, mounted here and nowhere else. Mounting is two lines — an
# import beside the two above and one call beside the ones below — and each lane adds its
# own pair when it lands. Uncomment yours:
#
#     from protein.db import cli as _db_cli
#     from protein.embed import cli as _esm_cli
#     from protein.search import cli as _search_cli
#     from protein.sifts import cli as _sifts_cli
#
# app.add_typer(_db_cli.app, name="db")
# app.add_typer(_esm_cli.app, name="esm")
# app.add_typer(_search_cli.app, name="search")
# app.add_typer(_sifts_cli.app, name="sifts")
#
# What a lane's own `cli.py` owes, all of it demonstrated above:
#
#   - a module-level `app = typer.Typer(help=..., no_args_is_help=True)`;
#   - every import aliased with a leading underscore, so `app.registered_commands` is the
#     whole of what the module exports;
#   - command names given explicitly and hyphenated -- `@app.command("table-row")` -- while
#     the function stays snake_case;
#   - a runnable doctest in the module docstring listing its own commands, which is what
#     makes a command added without a name fail the gate;
#   - `--json` on every command, rendered from a result dataclass's `as_json()` so the text
#     and the JSON cannot drift;
#   - one module-level `_<TOPIC>_ERRORS` tuple naming what the sub-app catches, spelled
#     once with a comment saying which failure is which exception;
#   - a failure that prints `error: {err}` to stderr and raises `typer.Exit(code=1)`.
