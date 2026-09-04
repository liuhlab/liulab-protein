"""A3M, the one alignment format this package reads and writes.

A3M is FASTA with two rules on top. **Case is the match state**: an uppercase residue or a
``-`` occupies a column, and a lowercase residue is an insertion that occupies none. And a
leading ``#`` line may carry the chain layout of a complex, which is where ColabFold writes
one.

biotite's record layer is adopted for reading, as in :mod:`protein.io.fasta`, and not for
writing: ``FastaFile.write_iter`` wraps a record over several lines where an A3M row is one
line. ``FastaFile.read_iter`` also drops a leading ``#``, so the comment is taken off the
handle here before biotite is given it.

Nothing here checks the shape of an alignment. That is
:class:`protein.msa.MSA`'s job, so a parsed alignment and a generated one meet the same rule.

Examples
--------
>>> print(format_records([("query", "MKTAY"), ("hit", "MKTaAY")], comment="#5"), end="")
#5
>query
MKTAY
>hit
MKTaAY
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from biotite.sequence.io.fasta import FastaFile

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

__all__ = ["COMMENT_PREFIX", "format_records", "read_records", "write_records"]

#: What a leading comment line starts with. One such line is carried; ColabFold spells the
#: residue counts and the copy counts of a complex's chains there.
COMMENT_PREFIX = "#"


def read_records(source: str | Path) -> tuple[str | None, list[tuple[str, str]]]:
    """Read the A3M file at ``source`` into its comment line and its records.

    Case is preserved, so an insert column stays an insert column, and a header comes back
    byte-for-byte, so a ``key=`` taxonomy field survives.

    Parameters
    ----------
    source : str or pathlib.Path
        The A3M file.

    Returns
    -------
    comment : str or None
        The leading ``#`` line, the ``#`` included and the newline stripped, or ``None`` when
        the file opens with a record.
    records : list of tuple of (str, str)
        One ``(header, row)`` pair per record, in file order. The header is everything after
        ``>``; the row is one unwrapped string.

    Examples
    --------
    >>> comment, records = read_records("tests/data/colabfold_pair.a3m")  # doctest: +SKIP
    >>> records[0]  # doctest: +SKIP
    ('101', 'MKTAYIAKQRQISHFSRQLEER')
    """
    with Path(source).open(encoding="utf-8") as handle:
        first = handle.readline()
        if first.startswith(COMMENT_PREFIX):
            comment = first.rstrip("\n")
        else:
            comment = None
            handle.seek(0)
        return comment, list(FastaFile.read_iter(handle))


def format_records(records: Iterable[tuple[str, str]], *, comment: str | None = None) -> str:
    """Render ``records`` as A3M text, writing no file.

    One line per row and no wrapping, which is what makes a file read by
    :func:`read_records` come back byte-for-byte.

    Parameters
    ----------
    records : iterable of tuple of (str, str)
        ``(header, row)`` pairs. The header is written after ``>`` verbatim.
    comment : str, optional
        A leading comment line, written verbatim, so it carries its own ``#``.

    Returns
    -------
    str
        The alignment, each line newline-terminated.

    Examples
    --------
    >>> print(format_records([("a", "MKT"), ("b", "MKkT")]), end="")
    >a
    MKT
    >b
    MKkT
    """
    return "".join(_lines(records, comment))


def write_records(
    destination: str | Path,
    records: Iterable[tuple[str, str]],
    *,
    comment: str | None = None,
) -> None:
    """Write ``records`` to the A3M file at ``destination``, replacing what is there.

    Streaming: a generator is consumed one record at a time.

    Parameters
    ----------
    destination : str or pathlib.Path
        Where to write. The parent directory must exist; nothing here creates one.
    records : iterable of tuple of (str, str)
        ``(header, row)`` pairs, as :func:`format_records` takes.
    comment : str, optional
        A leading comment line, written verbatim.
    """
    with Path(destination).open("w", encoding="utf-8") as handle:
        handle.writelines(_lines(records, comment))


def _lines(records: Iterable[tuple[str, str]], comment: str | None) -> Iterator[str]:
    """Yield the newline-terminated lines of one A3M file."""
    if comment is not None:
        yield f"{comment}\n"
    for header, row in records:
        yield f">{header}\n{row}\n"
