"""What a search points at, how it is tuned, and what the query is called.

Four names no one **External tool** owns. MMseqs2 and Foldseek take the same **Database**, the
same knobs and the same query header, so a module named after one of them cannot say where the
other's search is pointed.

:class:`SearchTarget` is all a search asks of a **Database** — a path — and
:func:`database_path` resolves either one of those or a registered name, reaching over to
:mod:`protein.db`, which owns the database layout.

Examples
--------
>>> search_flags(sensitivity=1.0, threads=4)
['-s', '1.0', '--threads', '4']
>>> DEFAULT_QUERY_NAME
'query'
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "DEFAULT_QUERY_NAME",
    "SearchTarget",
    "database_path",
    "search_flags",
]

#: What the ``query`` column says when the searched **Protein** has no accession. A FASTA
#: record needs a header, and the alternative is a blank ``query`` column.
DEFAULT_QUERY_NAME = "query"


class SearchTarget(Protocol):
    """What a search needs of a **Database**: the path its **External tool** is pointed at.

    One read-only attribute, so satisfying it constrains nothing else about the class.

    :attr:`path` is the **ffindex prefix**, not the directory holding it: MMseqs2 and Foldseek
    take ``.../swissprot`` and find ``swissprot.index`` and the rest beside it themselves.
    """

    @property
    def path(self) -> Path:
        """The ffindex prefix to search against."""
        ...


def search_flags(
    *,
    sensitivity: float | None = None,
    evalue: float | None = None,
    max_seqs: int | None = None,
    threads: int | None = None,
    extra: Sequence[str] = (),
) -> list[str]:
    """Turn the named search knobs into the arguments both tools spell the same way.

    Only what was asked for is passed: an omitted knob leaves the tool's own default, which
    is not this package's to restate.

    Parameters
    ----------
    sensitivity : float, optional
        ``-s``. Lower is faster and finds less.
    evalue : float, optional
        ``-e``. Hits above it are not reported.
    max_seqs : int, optional
        ``--max-seqs``. How many targets per query pass the prefilter, which caps the hits.
    threads : int, optional
        ``--threads``. **Worth naming on a shared machine**: both tools default to every core.
    extra : sequence of str, optional
        Further arguments, appended unread.

    Returns
    -------
    list of str
        The arguments, in the order the parameters are listed, with ``extra`` last.

    Examples
    --------
    >>> search_flags(sensitivity=1.0, threads=4)
    ['-s', '1.0', '--threads', '4']
    >>> search_flags(extra=["--comp-bias-corr", "0"])
    ['--comp-bias-corr', '0']
    """
    named: list[tuple[str, float | int | None]] = [
        ("-s", sensitivity),
        ("-e", evalue),
        ("--max-seqs", max_seqs),
        ("--threads", threads),
    ]
    flags = [part for flag, value in named if value is not None for part in (flag, str(value))]
    return [*flags, *extra]


def database_path(database: SearchTarget | str) -> Path:
    """Return the ffindex prefix to search against, from a **Database** or a registered name.

    A :class:`SearchTarget` answers with its own :attr:`~SearchTarget.path`. A ``str`` is a
    registered name, which :func:`protein.db.database_path` resolves — including the rule
    that **the prefix inside the directory is not always the name**.

    Parameters
    ----------
    database : SearchTarget or str
        A **Database** to take the path from, or the name of a registered one.

    Returns
    -------
    pathlib.Path
        The ffindex prefix. Nothing is created and no completion record is read.

    Raises
    ------
    LookupError
        If a name has no directory, or a directory holds no ffindex database. The message
        names the registered names there are.

    Examples
    --------
    >>> from pathlib import Path
    >>> class Registered:
    ...     path = Path("/data/protein/db/swissprot/swissprot")
    >>> database_path(Registered())
    PosixPath('/data/protein/db/swissprot/swissprot')
    >>> database_path("swissprot")                                   # doctest: +SKIP
    PosixPath('.../protein/db/swissprot/swissprot')
    """
    if not isinstance(database, str):
        return Path(database.path)

    # Deferred: `protein.db` reaches liulab-genome, and a search handed a path pays nothing
    # for it. The module and not the function, so a test that re-points the layout is seen.
    from protein import db

    return db.database_path(database)
