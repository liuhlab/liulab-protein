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

:func:`align` is the module's verb for a set of sequences the caller already holds. It runs
MUSCLE through biotite and anchors what comes back with :meth:`MSA.compress`, which is what
turns a symmetric alignment into an A3M.

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
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Self, cast

from biotite.application.muscle import Muscle5App

from protein import seq
from protein.external import Muscle

if TYPE_CHECKING:
    from collections.abc import Iterable

    from protein.external import ExternalTool

__all__ = ["MSA", "InvalidAlignmentError", "align", "count_match_states"]

#: Offending positions the lowercase message lists before it counts the rest.
_MAX_LISTED = 5

#: What a gap is written as. A3M also spells an insert-column gap ``.``; nothing here
#: produces one, and :meth:`MSA.compress` reads only this.
_GAP = "-"

#: What MUSCLE needs before it will align anything.
_MIN_ALIGNED = 2


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
