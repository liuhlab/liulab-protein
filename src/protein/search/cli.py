"""The ``protein search`` sub-app — one sequence against one **Database**, from a shell.

A thin Typer wrapper over the lane: it builds a :class:`~protein.core.Protein` from what was
typed, makes one :meth:`~protein.search.mixin.SearchMixin.search` call and renders the frame.
Building a ``Protein`` rather than passing the string on is deliberate — the alphabet is
checked before an **External tool** is started, so a stray ``*`` fails here rather than
inside mmseqs.

``struct``, the Foldseek half, is not here yet: it searches with coordinates, which
``Structure`` and ``Chain`` hold and this command has none of. It is one command added
beside ``seq``, against :data:`_SEARCH_ERRORS` and :class:`_Hits` as they stand.

**The rows go to stdout, tab-separated, and the header to stderr**, so the output pipes:
``protein search seq ... | cut -f2`` is the list of targets. ``--json`` carries the same
answer for anything that would otherwise parse the table.

Examples
--------
>>> from protein.search.cli import app
>>> [command.name for command in app.registered_commands]
['seq']
"""

from __future__ import annotations

import json as _json
from dataclasses import dataclass as _dataclass
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Any as _Any

import typer

from protein import Protein as _Protein
from protein.search.mmseqs import DEFAULT_QUERY_NAME as _DEFAULT_QUERY_NAME

if _TYPE_CHECKING:
    import pandas as pd

#: What a search failure raises, and each already names its next action: a database name
#: nothing is registered under is a ``LookupError``; a sequence outside the alphabet is a
#: ``ValueError`` (``protein.seq.InvalidResidueError``); a missing binary and a tool that
#: exited non-zero are ``RuntimeError``s (``protein.external.ToolNotFoundError`` is one); and
#: a scratch directory that could not be written is an ``OSError``.
_SEARCH_ERRORS = (LookupError, OSError, RuntimeError, ValueError)

app = typer.Typer(
    help="Search a sequence or a structure against a registered database.",
    no_args_is_help=True,
)


@_dataclass(frozen=True, slots=True)
class _Hits:
    """One search's answer, in the shape both renderings are built from.

    Attributes
    ----------
    query : str
        What the query was called, which is what the ``query`` column reports.
    database : str
        The database searched, as it was named on the command line.
    columns : tuple of str
        The hit table's column names, in the tool's own order.
    rows : tuple of tuple
        One tuple per hit, holding plain Python values.
    """

    query: str
    database: str
    columns: tuple[str, ...]
    rows: tuple[tuple[_Any, ...], ...]

    @classmethod
    def of(cls, frame: pd.DataFrame, *, query: str, database: str) -> _Hits:
        """Build from a hit table, converting every cell to a plain Python value."""
        return cls(
            query=query,
            database=database,
            columns=tuple(str(name) for name in frame.columns),
            rows=tuple(
                tuple(_plain(value) for value in row)
                for row in frame.itertuples(index=False, name=None)
            ),
        )

    def as_json(self) -> dict[str, _Any]:
        """Return the answer as JSON-ready data, keyed the way the table is columned."""
        return {
            "query": self.query,
            "database": self.database,
            "columns": list(self.columns),
            "hits": [dict(zip(self.columns, row, strict=True)) for row in self.rows],
        }


def _plain(value: _Any) -> _Any:
    """Return ``value`` as a plain Python object, so :mod:`json` can write it.

    A frame's cells are numpy scalars, which ``json.dumps`` refuses. ``.item()`` is what
    numpy offers for exactly this, and a ``str`` — which the identifier columns hold — has no
    such method and needs none.
    """
    item = getattr(value, "item", None)
    return item() if callable(item) else value


@app.command("seq")
def search_seq(
    sequence: str = typer.Argument(
        ...,
        help="The residues to search with, one sequence. Checked against the amino-acid "
        "alphabet before mmseqs is started.",
    ),
    database: str = typer.Argument(
        ...,
        help="The name of a registered sequence database, e.g. 'swissprot'. A name nothing "
        "is registered under names the ones that are.",
    ),
    identifier: str | None = typer.Option(
        None,
        "--id",
        metavar="ID",
        help=f"Name the query; it is what the `query` column reports. Defaults to "
        f"'{_DEFAULT_QUERY_NAME}'.",
    ),
    sensitivity: float | None = typer.Option(
        None, "--sensitivity", "-s", help="mmseqs -s. Lower is faster and finds less."
    ),
    evalue: float | None = typer.Option(
        None, "--evalue", "-e", help="mmseqs -e. Hits above it are not reported."
    ),
    max_seqs: int | None = typer.Option(
        None, "--max-seqs", help="mmseqs --max-seqs, which caps the hits per query."
    ),
    threads: int | None = typer.Option(
        None,
        "--threads",
        help="mmseqs --threads. Worth naming on a shared machine: the default is every core.",
    ),
    json: bool = typer.Option(False, "--json", help="Emit JSON instead of plain text."),
) -> None:
    """Search one amino-acid sequence against a registered database with MMseqs2.

    The everyday way to ask what a sequence looks like, without writing Python. The hits go
    to stdout tab-separated so the output pipes, and the column names to stderr so the
    header does not land in what you piped. `--json` carries the same answer, with each hit
    keyed by column name.

    Identity is `pident`, a **percentage**. Foldseek's `fident` is the same quantity as a
    fraction, and the two are never renamed into each other — the column you have says which
    number you are reading.

    Exits with code 1 when nothing is registered under the database name, when the sequence
    holds something outside the amino-acid alphabet, when mmseqs is not installed, and when
    the search itself fails.
    """
    try:
        query = _Protein(sequence, id=identifier)
        frame = query.search(
            database,
            sensitivity=sensitivity,
            evalue=evalue,
            max_seqs=max_seqs,
            threads=threads,
        )
    except _SEARCH_ERRORS as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(code=1) from err

    hits = _Hits.of(frame, query=identifier or _DEFAULT_QUERY_NAME, database=database)
    if json:
        typer.echo(_json.dumps(hits.as_json()))
        return
    _report(hits)


def _report(hits: _Hits) -> None:
    """Print the hits as a tab-separated table, header on stderr and rows on stdout."""
    typer.echo("\t".join(hits.columns), err=True)
    for row in hits.rows:
        typer.echo("\t".join(str(value) for value in row))
    typer.echo(f"{len(hits.rows)} hits for {hits.query} in {hits.database}", err=True)
