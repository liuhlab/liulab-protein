"""The ``protein structure`` sub-app — fill the coordinate cache, and look inside an entry.

Two commands, and each is here because the library alone is awkward from a shell.

``fetch`` is **the one step in this lane that needs the network**, so it is the one a person
runs on a login node before a job starts: the lab's compute nodes have none. ``show`` is how
you read an entry's chains without writing Python.

Searching is not here. ``protein search struct`` is beside ``protein search seq``, in
:mod:`protein.search.cli`, because what varies between the two is the query and not the
lane.

Examples
--------
>>> from protein.structure.cli import app
>>> [command.name for command in app.registered_commands]
['fetch', 'show']
"""

from __future__ import annotations

import json as _json
from dataclasses import dataclass as _dataclass
from pathlib import Path as _Path
from typing import Any as _Any

import typer

from protein.structure.structure import Structure as _Structure
from protein.structure.structure import fetch as _fetch

#: What these two commands catch. Every one of them already names its own next action, so
#: each becomes ``error: <message>`` and exit code 1 rather than a traceback.
_STRUCTURE_ERRORS = (LookupError, OSError, RuntimeError, ValueError)

app = typer.Typer(
    help="Cache a PDB entry's coordinates, and say what is in one.",
    no_args_is_help=True,
)


@_dataclass(frozen=True, slots=True)
class _Cached:
    """Where one entry's coordinates ended up.

    Attributes
    ----------
    id : str
        The entry id as it was typed.
    path : pathlib.Path
        The cached file.
    bytes : int
        Its size.
    """

    id: str
    path: _Path
    bytes: int

    def as_json(self) -> dict[str, _Any]:
        """Return this result as the mapping ``--json`` prints."""
        return {"id": self.id, "path": str(self.path), "bytes": self.bytes}


@_dataclass(frozen=True, slots=True)
class _Chains:
    """One structure's chains, in the shape both renderings are built from.

    Attributes
    ----------
    id : str
        What the structure is called.
    path : pathlib.Path
        The file it was read from.
    columns : tuple of str
        The row headings, in order.
    rows : tuple of tuple
        One tuple per chain, holding plain Python values.
    """

    id: str
    path: _Path
    columns: tuple[str, ...]
    rows: tuple[tuple[_Any, ...], ...]

    @classmethod
    def of(cls, structure: _Structure) -> _Chains:
        """Read every chain of ``structure``, and what is known about each."""
        rows = tuple(
            (
                chain.chain_id,
                chain.kind,
                len(chain.sequence) if chain.kind == "protein" else None,
                len(chain),
                ",".join(chain.uniprot),
            )
            for chain in structure.chains
        )
        return cls(
            id=structure.id,
            path=structure.path,
            columns=("chain", "kind", "residues", "atoms", "uniprot"),
            rows=rows,
        )

    def as_json(self) -> dict[str, _Any]:
        """Return the answer as JSON-ready data, keyed the way the table is columned."""
        return {
            "id": self.id,
            "path": str(self.path),
            "columns": list(self.columns),
            "chains": [dict(zip(self.columns, row, strict=True)) for row in self.rows],
        }


@app.command("fetch")
def fetch_command(
    pdb_id: str = typer.Argument(
        ...,
        metavar="PDB_ID",
        help="A PDB entry id, e.g. '1UBQ'. The cache is keyed lower-case.",
    ),
    json: bool = typer.Option(False, "--json", help="Emit JSON instead of plain text."),
) -> None:
    """Put one PDB entry's coordinates in the cache, downloading them if they are not there.

    Run it where there is a network. Already cached, it downloads nothing and reports the
    file that is there.

    Exits with code 1 when the entry is neither cached nor reachable.
    """
    try:
        path = _fetch(pdb_id)
    except _STRUCTURE_ERRORS as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(code=1) from err
    _render(_Cached(id=pdb_id, path=path, bytes=path.stat().st_size).as_json(), json=json)


@app.command("show")
def show_command(
    structure: str = typer.Argument(
        ...,
        help="A coordinate file, or a PDB entry id to read from the cache — fetching it if "
        "it is not there.",
    ),
    json: bool = typer.Option(False, "--json", help="Emit JSON instead of plain text."),
) -> None:
    """List a structure's chains: what each is, how big it is, and whose protein it is.

    The rows go to stdout tab-separated and the header to stderr, so the output pipes. An
    empty `uniprot` cell is a real answer — a nucleic-acid chain, a ligand chain, or an
    entry SIFTS never curated.

    Exits with code 1 when the coordinates cannot be read and when the SIFTS map is not
    prepared here; the message names the command that prepares it.
    """
    try:
        chains = _Chains.of(_open(structure))
    except _STRUCTURE_ERRORS as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(code=1) from err

    if json:
        typer.echo(_json.dumps(chains.as_json()))
        return
    typer.echo("\t".join(chains.columns), err=True)
    for row in chains.rows:
        typer.echo("\t".join("" if value is None else str(value) for value in row))
    typer.echo(f"{len(chains.rows)} chains in {chains.id} ({chains.path})", err=True)


def _open(structure: str) -> _Structure:
    """Return the structure ``structure`` names: an existing file, else a PDB entry id.

    The path is tried first because a file that exists is not an id somebody meant.
    """
    path = _Path(structure)
    if path.exists():
        return _Structure.from_file(path)
    return _Structure(structure)


def _render(fields: dict[str, _Any], *, json: bool) -> None:
    """Print one result, as JSON or as one ``key: value`` line each."""
    if json:
        typer.echo(_json.dumps(fields))
        return
    for key, value in fields.items():
        typer.echo(f"{key}: {value}")
