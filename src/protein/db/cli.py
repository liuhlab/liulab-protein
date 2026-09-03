"""The ``protein db`` sub-app — say what is registered, point at one, or fetch one.

``adopt`` and ``download`` bring a database into the registry, ``list`` and ``status``
report. None of the four changes a database, because **these databases are immutable**: the
index holds byte offsets into the data file, so editing the data breaks every offset. There
is deliberately no ``remove``, no ``rebuild`` and no ``index``.

``adopt`` is the one to reach for on a cluster, where the files are usually already there.

Examples
--------
>>> from protein.db.cli import app
>>> [command.name for command in app.registered_commands]
['list', 'adopt', 'download', 'status']
"""

from __future__ import annotations

import json as _json
from dataclasses import dataclass as _dataclass
from pathlib import Path as _Path
from typing import Annotated as _Annotated
from typing import Any as _Any

import typer

from protein import db as _db

#: What these four commands catch and report as ``error: ...`` with exit 1, rather than as a
#: traceback.
_DATABASE_ERRORS = (LookupError, OSError, RuntimeError, ValueError)

#: `Annotated` rather than a `typer` call in the default, which ruff's B008 refuses for a
#: `Path`-annotated parameter.
_PathArgument = _Annotated[_Path, typer.Argument(help="The database's directory on disk.")]

app = typer.Typer(
    help="Registered sequence and structure databases: list, adopt, download, status.",
    no_args_is_help=True,
)


@_dataclass(frozen=True, slots=True)
class _Listed:
    """One row of ``protein db list``.

    Attributes
    ----------
    name : str
        The name that addresses it.
    directory : str
        Where it lives, whether or not it is there.
    registered : bool
        Whether a completion record says it finished.
    declared : bool
        Whether ``protein db download <name>`` knows a source for it.
    description : str or None
        What the declaration says it is, when there is one.
    completed_at : str or None
        When registration finished.
    """

    name: str
    directory: str
    registered: bool
    declared: bool
    description: str | None = None
    completed_at: str | None = None

    def as_json(self) -> dict[str, _Any]:
        """Return this row as the mapping ``--json`` prints."""
        return {
            "name": self.name,
            "directory": self.directory,
            "registered": self.registered,
            "declared": self.declared,
            "description": self.description,
            "completed_at": self.completed_at,
        }


@app.command("list")
def list_command(
    json: _Annotated[bool, typer.Option("--json", help="Emit JSON instead of plain text.")] = False,
) -> None:
    """List every database registered here, and every one this package can fetch.

    A declared name with no directory is shown too: on a fresh machine that list is the
    answer to "what can I download?".
    """
    try:
        rows = _rows()
    except _DATABASE_ERRORS as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(code=1) from err

    if json:
        typer.echo(_json.dumps({"databases": [row.as_json() for row in rows]}))
        return
    if not rows:
        typer.echo(f"nothing registered under {_db.database_data_dir()}")
        return
    for row in rows:
        mark = "registered" if row.registered else "not registered"
        typer.echo(f"{row.name}\t{mark}\t{row.description or row.directory}")


@app.command("adopt")
def adopt_command(
    name: _Annotated[str, typer.Argument(help="The name to register it under, a slug.")],
    path: _PathArgument,
    kind: _Annotated[
        str | None,
        typer.Option("--kind", help="sequence or structure, for an undeclared name."),
    ] = None,
    force: _Annotated[
        bool, typer.Option("--force", help="Write a fresh record over an existing one.")
    ] = False,
    json: _Annotated[bool, typer.Option("--json", help="Emit JSON instead of plain text.")] = False,
) -> None:
    """Register a database that is already on disk, without copying or downloading it.

    Points ``<data dir>/db/<name>`` at ``path`` — the same directory, or a symlink to it —
    and writes the completion record beside the real files. Both a GPU-encoded database and
    a plain one are accepted; ``status`` says which this is.
    """
    try:
        database = _db.database_class(name, kind=kind).adopt(name, path, force=force)
        found = database.status()
    except _DATABASE_ERRORS as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(code=1) from err
    _render(found, json=json)


@app.command("download")
def download_command(
    name: _Annotated[str, typer.Argument(help="The name to register it under, a slug.")],
    source: _Annotated[
        str | None,
        typer.Option("--source", help="The tool's own spelling, e.g. UniProtKB/Swiss-Prot."),
    ] = None,
    kind: _Annotated[
        str | None,
        typer.Option("--kind", help="sequence or structure, for an undeclared name."),
    ] = None,
    force: _Annotated[
        bool, typer.Option("--force", help="Fetch again even when this name is registered.")
    ] = False,
    json: _Annotated[bool, typer.Option("--json", help="Emit JSON instead of plain text.")] = False,
) -> None:
    """Fetch a database with the tool's own downloader, then register what it left.

    **This package does not manage the download.** ``mmseqs databases`` and ``foldseek
    databases`` do the fetching; what is added here is the record. It needs a network, so it
    belongs on a login node, and the tool's own progress is left streaming to the terminal
    rather than captured.
    """
    try:
        database = _db.database_class(name, kind=kind).download(name, source=source, force=force)
        found = database.status()
    except _DATABASE_ERRORS as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(code=1) from err
    _render(found, json=json)


@app.command("status")
def status_command(
    name: _Annotated[str, typer.Argument(help="A registered name.")],
    kind: _Annotated[
        str | None,
        typer.Option("--kind", help="sequence or structure, for an undeclared name."),
    ] = None,
    json: _Annotated[bool, typer.Option("--json", help="Emit JSON instead of plain text.")] = False,
) -> None:
    """Say what is on disk for one database, without touching the network.

    Two entry counts are printed and each names the file it came from: ``index_entries`` is
    the searchable set and ``lookup_entries`` is every named entry.
    """
    try:
        found = _db.open_database(name, kind=kind).status()
    except _DATABASE_ERRORS as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(code=1) from err
    _render(found, json=json)


def _rows() -> list[_Listed]:
    """Return one row per registered directory, plus every declared name with none."""
    from genome.store import completion

    root = _db.database_data_dir()
    names = sorted({*_db.registered_names(), *_db.DECLARED})
    rows: list[_Listed] = []
    for name in names:
        declared = _db.DECLARED.get(name)
        record = completion.read_record(root / name)
        rows.append(
            _Listed(
                name=name,
                directory=str(root / name),
                registered=record is not None,
                declared=declared is not None,
                description=declared.description if declared is not None else None,
                completed_at=record.completed_at if record is not None else None,
            )
        )
    return rows


def _render(found: _db.DatabaseStatus, *, json: bool) -> None:
    """Print one status, as JSON or as one ``key: value`` line each."""
    if json:
        typer.echo(_json.dumps(found.as_json()))
        return
    for key, value in found.as_json().items():
        typer.echo(f"{key}: {value}")
