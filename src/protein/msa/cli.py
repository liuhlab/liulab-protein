"""The ``protein msa`` sub-app — one alignment, written where the caller said.

Two commands over one lane, differing in where the sequences come from. ``search`` builds a
:class:`~protein.core.Protein` from what was typed, so the alphabet is checked before an
**External tool** is started, and searches a **Database**; ``align`` reads a FASTA the caller
already has and lines it up with MUSCLE.

**The output path is an argument and not an option**, for both: nothing durable lands
anywhere the caller did not name. ``--json`` carries the same answer for anything that would
otherwise read the printed lines.

Examples
--------
>>> from protein.msa.cli import app
>>> [command.name for command in app.registered_commands]
['search', 'align']
"""

from __future__ import annotations

import json as _json
from dataclasses import dataclass as _dataclass
from pathlib import Path as _Path
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Annotated as _Annotated
from typing import Any as _Any

import typer

from protein import msa as _msa
from protein.core import Protein as _Protein
from protein.search.target import DEFAULT_QUERY_NAME as _DEFAULT_QUERY_NAME

if _TYPE_CHECKING:
    from collections.abc import Sequence

    from protein.msa.msa import MSA

#: What the two commands catch. Each already names its own next action, so it becomes
#: ``error: <message>`` and exit code 1 rather than a traceback.
_MSA_ERRORS = (LookupError, OSError, RuntimeError, ValueError)

app = typer.Typer(
    help="Build a multiple sequence alignment and write it as A3M.",
    no_args_is_help=True,
)


@_dataclass(frozen=True, slots=True)
class _Written:
    """One alignment, and the file it was kept in.

    Attributes
    ----------
    query : str
        Row 0's header — what the alignment is anchored on.
    depth : int
        How many rows it holds, the query included.
    match_states : int
        How many columns every row occupies.
    path : pathlib.Path
        The A3M written.
    """

    query: str
    depth: int
    match_states: int
    path: _Path

    @classmethod
    def of(cls, alignment: MSA, path: _Path) -> _Written:
        """Read what is worth reporting about one alignment written to ``path``."""
        return cls(
            query=alignment.query_header,
            depth=alignment.depth,
            match_states=alignment.match_states,
            path=path,
        )

    def as_json(self) -> dict[str, _Any]:
        """Return this result as the mapping ``--json`` prints."""
        return {
            "query": self.query,
            "depth": self.depth,
            "match_states": self.match_states,
            "path": str(self.path),
        }


# `Annotated` rather than a `typer` call in the default, which ruff's B008 refuses for a
# `Path`-annotated parameter.
@app.command("search")
def search_command(
    sequence: _Annotated[
        str,
        typer.Argument(
            help="The residues to align around, one sequence. Checked against the amino-acid "
            "alphabet before mmseqs is started."
        ),
    ],
    database: _Annotated[
        str,
        typer.Argument(
            help="The name of a registered sequence database. Nothing is shipped or adopted "
            "behind it, and a shallow set standing in for a deep one is a wrong answer."
        ),
    ],
    out: _Annotated[
        _Path,
        typer.Argument(
            help="Where to write the alignment, as A3M. Required: nothing durable lands "
            "anywhere the caller did not name."
        ),
    ],
    identifier: _Annotated[
        str | None,
        typer.Option(
            "--id",
            metavar="ID",
            help=f"Name the query; it is the header of row 0. Defaults to '{_DEFAULT_QUERY_NAME}'.",
        ),
    ] = None,
    sensitivity: _Annotated[
        float | None,
        typer.Option("--sensitivity", "-s", help="mmseqs -s. Lower is faster and finds less."),
    ] = None,
    evalue: _Annotated[
        float | None,
        typer.Option("--evalue", "-e", help="mmseqs -e. Hits above it are not reported."),
    ] = None,
    max_seqs: _Annotated[
        int | None,
        typer.Option("--max-seqs", help="mmseqs --max-seqs, which caps the hits per query."),
    ] = None,
    threads: _Annotated[
        int | None,
        typer.Option(
            "--threads",
            help="mmseqs --threads. Worth naming on a shared machine: the default is every core.",
        ),
    ] = None,
    json: _Annotated[bool, typer.Option("--json", help="Emit JSON instead of plain text.")] = False,
) -> None:
    """Search a database with one sequence and write the alignment it found.

    The depth is what the search found and no floor is enforced; a search that matched
    nothing writes the query alone. Each row's header is carried whole and gains
    `key=<organism id>` wherever it names one, which is what pairs the chains of a complex.

    Exits with code 1 when the database name is not registered, when the sequence holds
    something outside the amino-acid alphabet, when mmseqs is not installed, and when the
    search itself fails.
    """
    try:
        alignment = _Protein(sequence, id=identifier).msa(
            database,
            sensitivity=sensitivity,
            evalue=evalue,
            max_seqs=max_seqs,
            threads=threads,
        )
        written = _Written.of(alignment, alignment.write(out))
    except _MSA_ERRORS as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(code=1) from err
    _render({"database": database, **written.as_json()}, json=json)


@app.command("align")
def align_command(
    fasta: _Annotated[
        _Path,
        typer.Argument(help="A FASTA file of the ungapped sequences to line up, two or more."),
    ],
    out: _Annotated[
        _Path,
        typer.Argument(
            help="Where to write the alignment, as A3M. Required: nothing durable lands "
            "anywhere the caller did not name."
        ),
    ],
    query: _Annotated[
        str,
        typer.Option(
            "--query",
            metavar="HEADER",
            help="Which sequence to anchor on: its header, or the identifier that header "
            "opens with. It becomes row 0.",
        ),
    ],
    json: _Annotated[bool, typer.Option("--json", help="Emit JSON instead of plain text.")] = False,
) -> None:
    """Align a FASTA of sequences with MUSCLE, anchored on the one `--query` names.

    No database is involved. Headers are carried whole, so an `OX=` or `key=` taxonomy field
    reaches the alignment rather than being cut off with the description.

    Exits with code 1 when the file cannot be read, when it holds fewer than two records,
    when `--query` names none of them, when a sequence holds something outside the
    amino-acid alphabet, and when muscle is not installed or the alignment itself fails.
    """
    try:
        records = _read(fasta)
        alignment = _msa.align(records, query=_anchor(records, query))
        written = _Written.of(alignment, alignment.write(out))
    except _MSA_ERRORS as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(code=1) from err
    _render({"sequences": str(fasta), **written.as_json()}, json=json)


def _read(source: _Path) -> list[tuple[str, str]]:
    """Return the ``(header, residues)`` pairs of the FASTA at ``source``."""
    from protein.io import fasta

    return list(fasta.read_records(source))


def _anchor(records: Sequence[tuple[str, str]], wanted: str) -> str:
    """Return the header ``wanted`` designates, which may be the identifier it opens with.

    A UniProt header is a sentence, and nobody types one at a shell. An identifier naming
    exactly one record stands in for its header; anything else is handed on as it was typed,
    so :func:`protein.msa.muscle.align` raises the one error that names the headers there
    are.
    """
    from protein.io import fasta

    headers = [header for header, _ in records]
    if wanted in headers:
        return wanted
    matched = [header for header in headers if fasta.split_header(header)[0] == wanted]
    return matched[0] if len(matched) == 1 else wanted


def _render(fields: dict[str, _Any], *, json: bool) -> None:
    """Print one result, as JSON or as one ``key: value`` line each."""
    if json:
        typer.echo(_json.dumps(fields))
        return
    for key, value in fields.items():
        typer.echo(f"{key}: {value}")
