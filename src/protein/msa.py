"""The :class:`MSA` class — one query-anchored alignment, held as text.

An **MSA** is ``(header, row)`` pairs of plain :class:`str`, in file order, with row 0 the
query. **Case is the match state**, so nothing here uppercases anything and nothing holds
biotite's ``Alignment``, which uppercases on construction and renders every gap as ``-``.
Plain strings rather than typed sequences for a second reason: a row is gapped, so it falls
outside :data:`protein.seq.ALPHABET`, and the same class has to be able to hold a nucleic
alignment without lying about its alphabet.

The **shape** is checked at construction and the **residues are not**. An alignment is a
file's content, and refusing a row because a database entry spells ``U`` would reject
well-formed A3M this class only holds and hands back.

Two verbs, because the two jobs have different shapes. :func:`search` searches a **Database**
and is what :meth:`protein.core.Protein.msa` calls; :func:`align` takes a set of sequences
the caller already holds, runs MUSCLE through biotite, and anchors the symmetric alignment
that comes back with :meth:`MSA.compress`.

Examples
--------
>>> from protein import MSA
>>> msa = MSA([("query", "MKTAY"), ("hit", "MKTaAY")], comment="#5")
>>> msa
MSA(depth 2, 5 match states)
>>> msa.query
'MKTAY'
>>> print(msa.to_a3m(), end="")
#5
>query
MKTAY
>hit
MKTaAY
>>> from protein.msa import app
>>> [command.name for command in app.registered_commands]
['search', 'align']
"""

from __future__ import annotations

import json as _json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Self, cast

import typer
from biotite.application.muscle import Muscle5App

from protein import seq
from protein.core import Protein
from protein.external import Mmseqs, Muscle
from protein.search.target import DEFAULT_QUERY_NAME, database_path, search_flags

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from protein.external import ExternalTool, MmseqsLikeTool
    from protein.search.target import SearchTarget

__all__ = [
    "MSA",
    "InvalidAlignmentError",
    "align",
    "app",
    "count_match_states",
    "organism_id",
    "search",
    "with_organism_key",
]

#: Offending positions the lowercase message lists before it counts the rest.
_MAX_LISTED = 5

#: What a gap is written as. A3M also spells an insert-column gap ``.``; nothing here
#: produces one, and :meth:`MSA.compress` reads only this.
_GAP = "-"

#: What MUSCLE needs before it will align anything.
_MIN_ALIGNED = 2

#: How a search's own header spells the organism, and what a folding tool reads instead.
#: UniProtKB writes ``OX=``, UniRef writes ``TaxID=``, and a header that has already been
#: through :func:`with_organism_key` carries ``key=``.
_ORGANISM_ID = re.compile(r"\b(?:OX|TaxID|key)=(\d+)\b")

#: What ``result2msa`` is asked to write. The A3M modes replace a hit's header with its
#: accession alone, which throws away the organism id that pairs the chains of a complex;
#: this one keeps the header whole. What it costs is the hit's own insertions, which this
#: writer drops — every row it writes is one column per query residue, which is a valid A3M
#: with no insert columns.
_MSA_FORMAT_MODE = 2

#: What the unpacked alignment is named. ``0`` is ``--unpack-name-mode``: one query, so one
#: file, named by its database key rather than through a ``.lookup`` the query database is
#: not obliged to have.
_A3M_SUFFIX = ".a3m"
_UNPACK_BY_KEY = 0


class InvalidAlignmentError(ValueError):
    """Raised when rows do not form a query-anchored A3M.

    Its own class rather than a bare :class:`ValueError`, so a caller can tell a malformed
    alignment from every other way a constructor argument can be wrong. It reports shape and
    never residues.

    Attributes
    ----------
    row : int or None
        The zero-based index of the offending row, or ``None`` where the alignment as a whole
        is at fault.
    """

    def __init__(self, message: str, *, row: int | None = None) -> None:
        """Build the error from its message and, where there is one, the row to blame."""
        self.row = row
        super().__init__(message)


def count_match_states(row: str) -> int:
    """Return how many match states one A3M row occupies.

    Every character that is not a lowercase letter: an uppercase residue and a ``-`` each
    occupy a column, a lowercase residue is an insertion and occupies none. Every row of an
    alignment answers the same number, which is what :class:`MSA` checks.

    Parameters
    ----------
    row : str
        One A3M row, as read.

    Returns
    -------
    int
        The match-state count.

    Examples
    --------
    >>> count_match_states("MKTAY")
    5
    >>> count_match_states("MKTaaAY")
    5
    >>> count_match_states("MK--Y")
    5
    """
    return sum(1 for character in row if not character.islower())


class MSA:
    """One multiple sequence alignment in A3M, anchored on the query in row 0.

    Parameters
    ----------
    rows : iterable of tuple of (str, str)
        ``(header, row)`` pairs, in file order. Row 0 is the query and its residues define
        the match states. Case arrives as it is and is never changed.
    comment : str, optional
        A leading comment line, carried verbatim and so carrying its own ``#``. ColabFold
        writes the chain layout of a complex there, and biotite's FASTA reader drops it.

    Attributes
    ----------
    rows : tuple of tuple of (str, str)
        The pairs, in file order.
    comment : str or None
        The comment line, or ``None`` when there is none.

    Raises
    ------
    InvalidAlignmentError
        If there is no row 0, if row 0 carries lowercase — which would mean the alignment is
        anchored on nothing — or if two rows disagree on their match-state count.

    Examples
    --------
    >>> msa = MSA([("query", "MKTAY"), ("hit", "MKTaAY")])
    >>> msa.depth, msa.match_states, len(msa)
    (2, 5, 2)
    >>> msa.query_header
    'query'
    >>> MSA([("query", "MKTaY")])
    Traceback (most recent call last):
        ...
    protein.msa.InvalidAlignmentError: row 0 is the query and carries lowercase at 3
    """

    def __init__(self, rows: Iterable[tuple[str, str]], *, comment: str | None = None) -> None:
        self.rows: tuple[tuple[str, str], ...] = tuple((header, row) for header, row in rows)
        self.comment = comment
        self._match_states = _check(self.rows)

    @classmethod
    def from_a3m(cls, path: str | Path) -> Self:
        """Read the A3M file at ``path``.

        Parameters
        ----------
        path : str or pathlib.Path
            The A3M file.

        Returns
        -------
        MSA
            Headers, case and the leading ``#`` line as they were on disk, so
            :meth:`to_a3m` gives the file back.

        Raises
        ------
        InvalidAlignmentError
            If the file is not a query-anchored A3M.

        Examples
        --------
        >>> MSA.from_a3m("tests/data/colabfold_pair.a3m")  # doctest: +SKIP
        MSA(depth 3, 22 match states)
        """
        from protein.io import a3m

        comment, records = a3m.read_records(path)
        return cls(records, comment=comment)

    def to_a3m(self) -> str:
        """Return this alignment as A3M text, writing no file.

        One line per row, so an alignment read from a file comes back byte-for-byte. This is
        the exit a tool taking a buffer wants; :meth:`write` is the one for a tool taking a
        path.

        Returns
        -------
        str
            The alignment, each line newline-terminated.

        Examples
        --------
        >>> print(MSA([("a", "MKT"), ("b", "MKkT")]).to_a3m(), end="")
        >a
        MKT
        >b
        MKkT
        """
        from protein.io import a3m

        return a3m.format_records(self.rows, comment=self.comment)

    def write(self, path: str | Path) -> Path:
        """Write this alignment to ``path`` as A3M, replacing what is there.

        The path is required and defaults nowhere: nothing durable lands in the **Data dir**
        without the caller saying where.

        Parameters
        ----------
        path : str or pathlib.Path
            Where to write. The parent directory must exist; nothing here creates one.

        Returns
        -------
        pathlib.Path
            The path written, so a call can be chained onto whatever takes the file.
        """
        from protein.io import a3m

        a3m.write_records(path, self.rows, comment=self.comment)
        return Path(path)

    def compress(self, index: int = 0) -> Self:
        """Anchor this alignment on row ``index`` and return the result.

        The columns where that row has a gap stop being columns: a residue in one becomes a
        lowercase insertion, and a gap in one disappears. The designated row leads the
        result, which is what makes it the query. There is no ``expand``: nothing in this
        package needs a rectangular matrix back.

        Parameters
        ----------
        index : int, default 0
            The row to anchor on. It is moved to row 0; the rest keep their order.

        Returns
        -------
        MSA
            A new alignment, checked at construction like any other. The comment is carried.

        Raises
        ------
        InvalidAlignmentError
            If the rows are not all as long as row ``index``. Compressing needs a symmetric
            alignment — one alignment column per character — and an A3M is not one.
        IndexError
            If there is no row ``index``.

        Examples
        --------
        >>> MSA([("query", "MK-TAY"), ("hit", "MKWTAY")]).compress().rows
        (('query', 'MKTAY'), ('hit', 'MKwTAY'))
        >>> MSA([("a", "MK-T"), ("b", "MKWT")]).compress(1).rows
        (('b', 'MKWT'), ('a', 'MK-T'))
        """
        anchor = self.rows[index][1]
        width = len(anchor)
        ragged = next((row for row, (_, text) in enumerate(self.rows) if len(text) != width), None)
        if ragged is not None:
            raise InvalidAlignmentError(
                f"compress anchors on row {index}, which needs every row to be {width} "
                f"characters long; row {ragged} is {len(self.rows[ragged][1])}. A symmetric "
                f"alignment has one character per column, and an A3M does not.",
                row=ragged,
            )
        order = [index, *(row for row in range(len(self.rows)) if row != index)]
        return type(self)(
            ((self.rows[row][0], _demote(self.rows[row][1], anchor)) for row in order),
            comment=self.comment,
        )

    @property
    def depth(self) -> int:
        """How many rows the alignment holds, the query included.

        Examples
        --------
        >>> MSA([("query", "MKTAY"), ("hit", "MKTaAY")]).depth
        2
        """
        return len(self.rows)

    @property
    def match_states(self) -> int:
        """How many columns every row occupies, counted once at construction.

        Examples
        --------
        >>> MSA([("query", "MK--Y"), ("hit", "MKTaaAY")]).match_states
        5
        """
        return self._match_states

    @property
    def query(self) -> str:
        """Row 0's residues — the sequence the alignment is anchored on.

        Examples
        --------
        >>> MSA([("query", "MKTAY")]).query
        'MKTAY'
        """
        return self.rows[0][1]

    @property
    def query_header(self) -> str:
        """Row 0's header, byte-for-byte as it was read.

        Examples
        --------
        >>> MSA([("101 key=9606", "MKTAY")]).query_header
        '101 key=9606'
        """
        return self.rows[0][0]

    def __len__(self) -> int:
        """Return the depth — the same number :attr:`depth` gives."""
        return len(self.rows)

    def __repr__(self) -> str:
        """Return e.g. ``MSA(depth 512, 214 match states)``.

        Examples
        --------
        >>> MSA([("query", "MKTAY")])
        MSA(depth 1, 5 match states)
        """
        return f"{type(self).__name__}(depth {self.depth}, {self.match_states} match states)"


def align(
    sequences: Mapping[str, str] | Iterable[tuple[str, str]],
    *,
    query: str,
    tool: ExternalTool | None = None,
) -> MSA:
    """Align sequences the caller already holds, and anchor the result on ``query``.

    A function and not a method: MUSCLE takes a **set**, and this package has no class for
    one. No **Database** is involved and none is searched — this is the verb for homologues
    that came from a paper, a colleague or an earlier search.

    biotite drives the binary. Its ``Muscle5App`` — version 5, not the ``MuscleApp`` that
    wraps version 3 — owns the temporary files, the arguments and the parsing; this package
    locates the binary, builds the sequences through :func:`protein.seq.to_protein_sequence`
    and hands both over. MUSCLE aligns symmetrically, so the result is anchored by
    :meth:`MSA.compress` before it is returned.

    Parameters
    ----------
    sequences : mapping of str to str, or iterable of tuple of (str, str)
        ``(header, residues)`` pairs, which is what :func:`protein.io.fasta.read_records`
        yields, or a mapping of the same. Residues are ungapped.
    query : str
        The header of the sequence to anchor on. It becomes row 0.
    tool : protein.external.ExternalTool, optional
        Where the binary is. Defaults to :class:`~protein.external.Muscle`.

    Returns
    -------
    MSA
        Query-anchored, in the order the sequences arrived except that the query leads.

    Raises
    ------
    ValueError
        If fewer than two sequences were given. MUSCLE aligns a set, not a sequence.
    LookupError
        If no sequence carries the ``query`` header. The message names the headers there are.
    protein.seq.InvalidResidueError
        If a sequence holds anything outside :data:`protein.seq.ALPHABET`.
    protein.external.ToolNotFoundError
        If ``muscle`` is not installed.

    Warns
    -----
    protein.seq.ResidueCoercionWarning
        If a sequence holds ``U``, ``O`` or ``J``, which this package folds to ``X``.

    Examples
    --------
    >>> msa = align({"P01308": "MKTAYIAK", "Q6YK33": "MKTYIAK"}, query="P01308")  # doctest: +SKIP
    >>> msa.query_header                                                          # doctest: +SKIP
    'P01308'
    """
    records = _records(sequences)
    if len(records) < _MIN_ALIGNED:
        raise ValueError(
            f"align takes at least {_MIN_ALIGNED} sequences and was given {len(records)}; "
            f"one sequence is not an alignment."
        )
    headers = [header for header, _ in records]
    if query not in headers:
        raise LookupError(
            f"no sequence is headed {query!r}, so there is nothing to anchor on. The "
            f"headers given are {headers}."
        )
    typed = [seq.to_protein_sequence(residues, name=header) for header, residues in records]
    located = tool if tool is not None else Muscle()
    alignment = Muscle5App.align(typed, bin_path=located.path)
    rows = zip(headers, alignment.get_gapped_sequences(), strict=True)
    return MSA(rows).compress(headers.index(query))


def search(
    sequence: str,
    database: SearchTarget | str,
    *,
    query_name: str = DEFAULT_QUERY_NAME,
    tool: MmseqsLikeTool | None = None,
    sensitivity: float | None = None,
    evalue: float | None = None,
    max_seqs: int | None = None,
    threads: int | None = None,
    extra: Sequence[str] = (),
) -> MSA:
    """Search ``database`` with one sequence and return the alignment it found.

    Four MMseqs2 invocations, each of them a single one: ``createdb`` for the query,
    ``search``, ``result2msa``, ``unpackdb``. The chaining is here rather than in
    :mod:`protein.external`, which owns the grammar and not the recipe.

    Everything the run makes lives and dies inside a
    :meth:`~protein.external.MmseqsLikeTool.scratch_dir`, so it leaves nothing behind and
    **there is no output path**: an alignment is a value, like a hit table.
    :meth:`MSA.write` is how one is kept.

    The headers a hit arrives under are carried whole and gain ``key=<organism id>`` wherever
    they name one — see :func:`with_organism_key` for why that matters.

    Parameters
    ----------
    sequence : str
        The residues to search with.
    database : protein.search.target.SearchTarget or str
        What to search against: a **Database**, or the name of a registered one. Required;
        nothing is shipped or adopted behind it.
    query_name : str, default "query"
        The FASTA header the query is written under, and the header of row 0.
    tool : protein.external.MmseqsLikeTool, optional
        The tool to drive. Defaults to :class:`~protein.external.Mmseqs`.
    sensitivity, evalue, max_seqs, threads : optional
        As :func:`protein.search.target.search_flags`. They reach the ``search`` verb.
    extra : sequence of str, optional
        Further arguments for ``search``, appended unread.

    Returns
    -------
    MSA
        Query-anchored, row 0 the query. Depth 1 — the query alone — when the search found
        nothing; a thin alignment is not refused.

    Raises
    ------
    LookupError
        If ``database`` names nothing registered.
    protein.external.ToolNotFoundError
        If ``mmseqs`` is not installed.
    RuntimeError
        If any of the four invocations exits non-zero.
    InvalidAlignmentError
        If what came back is not a query-anchored alignment.

    Examples
    --------
    >>> search("MKTAYIAKQRQISFVKSHFSRQ", "uniref30")           # doctest: +SKIP
    MSA(depth 1281, 22 match states)
    """
    from protein.io import fasta

    driver = tool if tool is not None else Mmseqs()
    target = database_path(database)
    flags = search_flags(
        sensitivity=sensitivity,
        evalue=evalue,
        max_seqs=max_seqs,
        threads=threads,
        extra=extra,
    )
    with driver.scratch_dir("msa") as work:
        query = work / "query.fasta"
        fasta.write_records(query, [(query_name, sequence)])
        query_db = driver.createdb([query], work / "querydb")
        result_db = driver.search(query_db, target, work / "result", extra=flags)
        msa_db = driver.result2msa(
            query_db, target, result_db, work / "msadb", format_mode=_MSA_FORMAT_MODE
        )
        unpacked = work / "unpacked"
        unpacked.mkdir()
        driver.unpackdb(msa_db, unpacked, suffix=_A3M_SUFFIX, name_mode=_UNPACK_BY_KEY)
        return _read_unpacked(unpacked, query_name, sequence)


def organism_id(header: str) -> int | None:
    """Return the NCBI organism id ``header`` names, or ``None`` when it names none.

    Three spellings, because the databases worth searching disagree: UniProtKB writes
    ``OX=``, UniRef writes ``TaxID=``, and a header already carrying ``key=`` answers with
    that.

    Parameters
    ----------
    header : str
        One FASTA header, as the search wrote it.

    Returns
    -------
    int or None
        The organism id.

    Examples
    --------
    >>> organism_id("sp|P01308|INS_HUMAN Insulin OS=Homo sapiens OX=9606 GN=INS")
    9606
    >>> organism_id("UniRef100_A0A0 Cluster: x n=2 Tax=Mus musculus TaxID=10090")
    10090
    >>> organism_id("101") is None
    True
    """
    found = _ORGANISM_ID.search(header)
    return int(found.group(1)) if found else None


def with_organism_key(header: str) -> str:
    """Return ``header`` with ``key=<organism id>`` on the end, where it names one.

    **A row without one is unpaired.** ESMFold2 pairs the chains of a complex by a
    ``key=<int>`` match over the FASTA header, and a chain whose rows carry no key folds
    block-diagonal with nothing raised — a wrong answer that looks like an answer. The
    original header stays in front byte-for-byte, so nothing it said is lost.

    Parameters
    ----------
    header : str
        One FASTA header, as the search wrote it.

    Returns
    -------
    str
        ``header`` unchanged when it names no organism, or already carries a key.

    Examples
    --------
    >>> with_organism_key("sp|P01315|INS_PIG Insulin OS=Sus scrofa OX=9823")
    'sp|P01315|INS_PIG Insulin OS=Sus scrofa OX=9823 key=9823'
    >>> with_organism_key("101 key=9606")
    '101 key=9606'
    >>> with_organism_key("101")
    '101'
    """
    if "key=" in header:
        return header
    found = organism_id(header)
    return header if found is None else f"{header} key={found}"


def _read_unpacked(directory: Path, query_name: str, sequence: str) -> MSA:
    """Return the one alignment ``unpackdb`` wrote, or the query alone when it wrote none."""
    from protein.io import a3m

    written = sorted(directory.glob(f"*{_A3M_SUFFIX}"))
    if not written:
        return MSA([(query_name, sequence)])
    comment, records = a3m.read_records(written[0])
    return MSA(((with_organism_key(header), row) for header, row in records), comment=comment)


def _records(sequences: Mapping[str, str] | Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    """Return the ``(header, residues)`` pairs, from a mapping or from pairs.

    The casts are the argument's shape and not a doubt about it: a ``Mapping`` is also an
    ``Iterable``, so no static reading can subtract one branch from the other.
    """
    if isinstance(sequences, Mapping):
        return list(cast("Mapping[str, str]", sequences).items())
    return list(cast("Iterable[tuple[str, str]]", sequences))


def _demote(row: str, anchor: str) -> str:
    """Return ``row`` with the columns where ``anchor`` has a gap demoted to insertions."""
    kept = [
        character.lower() if anchored == _GAP else character
        for character, anchored in zip(row, anchor, strict=True)
        if anchored != _GAP or character != _GAP
    ]
    return "".join(kept)


def _check(rows: tuple[tuple[str, str], ...]) -> int:
    """Return the match-state count every row shares, or raise :class:`InvalidAlignmentError`.

    Shape and never residues: what a row spells is the file's business, but how much of a
    column it occupies is the alignment's.
    """
    if not rows:
        raise InvalidAlignmentError(
            "an alignment is anchored on its query, so it holds at least row 0, and this "
            "one holds no rows at all."
        )
    query = rows[0][1]
    lowered = [index for index, character in enumerate(query) if character.islower()]
    if lowered:
        listed = ", ".join(str(index) for index in lowered[:_MAX_LISTED])
        rest = f", and {len(lowered) - _MAX_LISTED} more" if len(lowered) > _MAX_LISTED else ""
        raise InvalidAlignmentError(
            f"row 0 is the query and carries lowercase at {listed}{rest}", row=0
        )
    expected = count_match_states(query)
    for index, (_, row) in enumerate(rows):
        found = count_match_states(row)
        if found != expected:
            raise InvalidAlignmentError(
                f"row {index} occupies {found} match states where row 0 occupies {expected}; "
                f"every row of an A3M shares one count.",
                row=index,
            )
    return expected


#: What the two commands catch. Each already names its own next action, so it becomes
#: ``error: <message>`` and exit code 1 rather than a traceback.
_MSA_ERRORS = (LookupError, OSError, RuntimeError, ValueError)

app = typer.Typer(
    help="Build a multiple sequence alignment and write it as A3M.",
    no_args_is_help=True,
)


@dataclass(frozen=True, slots=True)
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
    path: Path

    @classmethod
    def of(cls, alignment: MSA, path: Path) -> _Written:
        """Read what is worth reporting about one alignment written to ``path``."""
        return cls(
            query=alignment.query_header,
            depth=alignment.depth,
            match_states=alignment.match_states,
            path=path,
        )

    def as_json(self) -> dict[str, Any]:
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
    sequence: Annotated[
        str,
        typer.Argument(
            help="The residues to align around, one sequence. Checked against the amino-acid "
            "alphabet before mmseqs is started."
        ),
    ],
    database: Annotated[
        str,
        typer.Argument(
            help="The name of a registered sequence database. Nothing is shipped or adopted "
            "behind it, and a shallow set standing in for a deep one is a wrong answer."
        ),
    ],
    out: Annotated[
        Path,
        typer.Argument(
            help="Where to write the alignment, as A3M. Required: nothing durable lands "
            "anywhere the caller did not name."
        ),
    ],
    identifier: Annotated[
        str | None,
        typer.Option(
            "--id",
            metavar="ID",
            help=f"Name the query; it is the header of row 0. Defaults to '{DEFAULT_QUERY_NAME}'.",
        ),
    ] = None,
    sensitivity: Annotated[
        float | None,
        typer.Option("--sensitivity", "-s", help="mmseqs -s. Lower is faster and finds less."),
    ] = None,
    evalue: Annotated[
        float | None,
        typer.Option("--evalue", "-e", help="mmseqs -e. Hits above it are not reported."),
    ] = None,
    max_seqs: Annotated[
        int | None,
        typer.Option("--max-seqs", help="mmseqs --max-seqs, which caps the hits per query."),
    ] = None,
    threads: Annotated[
        int | None,
        typer.Option(
            "--threads",
            help="mmseqs --threads. Worth naming on a shared machine: the default is every core.",
        ),
    ] = None,
    json: Annotated[bool, typer.Option("--json", help="Emit JSON instead of plain text.")] = False,
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
        alignment = Protein(sequence, id=identifier).msa(
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
    fasta: Annotated[
        Path,
        typer.Argument(help="A FASTA file of the ungapped sequences to line up, two or more."),
    ],
    out: Annotated[
        Path,
        typer.Argument(
            help="Where to write the alignment, as A3M. Required: nothing durable lands "
            "anywhere the caller did not name."
        ),
    ],
    query: Annotated[
        str,
        typer.Option(
            "--query",
            metavar="HEADER",
            help="Which sequence to anchor on: its header, or the identifier that header "
            "opens with. It becomes row 0.",
        ),
    ],
    json: Annotated[bool, typer.Option("--json", help="Emit JSON instead of plain text.")] = False,
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
        alignment = align(records, query=_anchor(records, query))
        written = _Written.of(alignment, alignment.write(out))
    except _MSA_ERRORS as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(code=1) from err
    _render({"sequences": str(fasta), **written.as_json()}, json=json)


def _read(source: Path) -> list[tuple[str, str]]:
    """Return the ``(header, residues)`` pairs of the FASTA at ``source``."""
    from protein.io import fasta

    return list(fasta.read_records(source))


def _anchor(records: Sequence[tuple[str, str]], wanted: str) -> str:
    """Return the header ``wanted`` designates, which may be the identifier it opens with.

    A UniProt header is a sentence, and nobody types one at a shell. An identifier naming
    exactly one record stands in for its header; anything else is handed on as it was typed,
    so :func:`align` raises the one error that names the headers there are.
    """
    from protein.io import fasta

    headers = [header for header, _ in records]
    if wanted in headers:
        return wanted
    matched = [header for header in headers if fasta.split_header(header)[0] == wanted]
    return matched[0] if len(matched) == 1 else wanted


def _render(fields: dict[str, Any], *, json: bool) -> None:
    """Print one result, as JSON or as one ``key: value`` line each."""
    if json:
        typer.echo(_json.dumps(fields))
        return
    for key, value in fields.items():
        typer.echo(f"{key}: {value}")
