"""Sequence search with MMseqs2, and the hit table both **External tool**s answer with.

:func:`search` is the whole lane: one sequence, one **Database**, one
``mmseqs easy-search``, one :class:`~pandas.DataFrame`.

**The hit table's grammar is here, not in each tool's module.** Foldseek vendors MMseqs2, so
:mod:`protein.search.foldseek` reads its results with :func:`read_hits` from here rather than
with a second parser, and :data:`COLUMN_DTYPES` names and types every column either tool can
emit exactly once.

The two tools disagree about identity on purpose: MMseqs2 reports ``pident``, a percentage,
and Foldseek reports ``fident``, a fraction. Nothing here renames one to the other — a frame
says which number it carries by which column it has.

A bare name is resolved by :mod:`protein.db`, which owns the database layout;
:func:`database_path` is the one line that reaches over.

Examples
--------
>>> from protein.external import Foldseek, Mmseqs
>>> Mmseqs().format_columns[:3]
('query', 'target', 'pident')
>>> hit_dtypes(Mmseqs())["pident"], hit_dtypes(Foldseek())["fident"]
('float64', 'float64')
>>> empty_hits(Mmseqs()).shape
(0, 12)
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

from protein.external import Mmseqs

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pandas as pd

    from protein.external import MmseqsLikeTool

__all__ = [
    "COLUMN_DTYPES",
    "DEFAULT_QUERY_NAME",
    "SearchTarget",
    "database_path",
    "empty_hits",
    "hit_dtypes",
    "read_hits",
    "search",
    "search_flags",
]

#: Every column either **External tool** can be asked for, mapped to the dtype it is read
#: as — one table rather than one per tool. ``pident`` is a percentage and ``fident`` the
#: same quantity as a fraction; nothing here converts one into the other.
COLUMN_DTYPES: Mapping[str, str] = MappingProxyType(
    {
        "query": "string",
        "target": "string",
        "pident": "float64",
        "fident": "float64",
        "alnlen": "int64",
        "mismatch": "int64",
        "gapopen": "int64",
        "qstart": "int64",
        "qend": "int64",
        "tstart": "int64",
        "tend": "int64",
        "evalue": "float64",
        "bits": "float64",
        "alntmscore": "float64",
        "lddt": "float64",
    }
)

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


def hit_dtypes(tool: MmseqsLikeTool) -> dict[str, str]:
    """Return ``tool``'s result columns, in order, mapped to the dtype each is read as.

    Parameters
    ----------
    tool : protein.external.MmseqsLikeTool
        The tool whose ``--format-output`` the columns come from.

    Returns
    -------
    dict of str to str
        :attr:`~protein.external.MmseqsLikeTool.format_columns` against
        :data:`COLUMN_DTYPES`, in the tool's own order.

    Raises
    ------
    KeyError
        If the tool asks for a column :data:`COLUMN_DTYPES` does not type, rather than
        producing a frame whose dtype pandas guessed.

    Examples
    --------
    >>> from protein.external import Foldseek
    >>> list(hit_dtypes(Foldseek()))[-2:]
    ['alntmscore', 'lddt']
    """
    unknown = [name for name in tool.format_columns if name not in COLUMN_DTYPES]
    if unknown:
        raise KeyError(
            f"{tool.name} asks for {unknown}, which protein.search.mmseqs.COLUMN_DTYPES does "
            f"not type. Add each one there, so every column is named and typed in one place."
        )
    return {name: COLUMN_DTYPES[name] for name in tool.format_columns}


def empty_hits(tool: MmseqsLikeTool) -> pd.DataFrame:
    """Return a hit table of no rows, with the columns and dtypes a full one has.

    What a search that found nothing answers with, so a caller never has to branch on
    emptiness to learn the schema.

    Parameters
    ----------
    tool : protein.external.MmseqsLikeTool
        The tool whose columns the empty table carries.

    Returns
    -------
    pandas.DataFrame
        Zero rows, :func:`hit_dtypes`' columns and dtypes.

    Examples
    --------
    >>> from protein.external import Mmseqs
    >>> frame = empty_hits(Mmseqs())
    >>> frame.shape, frame.dtypes["evalue"]
    ((0, 12), dtype('float64'))
    """
    import pandas as pd

    return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in hit_dtypes(tool).items()})


def read_hits(output: Path, tool: MmseqsLikeTool) -> pd.DataFrame:
    """Read what ``tool`` wrote to ``output`` as a named, typed hit table.

    The tab-separated file carries no header, and this is the one place the names are put
    back on. An output that is missing or empty is a search that found nothing.

    Parameters
    ----------
    output : pathlib.Path
        The tab-separated hits an ``easy-search`` wrote.
    tool : protein.external.MmseqsLikeTool
        The tool that wrote it, whose ``--format-output`` named the columns.

    Returns
    -------
    pandas.DataFrame
        One row per hit, :func:`hit_dtypes`' columns in the tool's order.

    Examples
    --------
    >>> from pathlib import Path
    >>> from protein.external import Mmseqs
    >>> hits = read_hits(Path("tests/data/mmseqs_hits_p01308.tsv"), Mmseqs())  # doctest: +SKIP
    >>> hits.loc[0, "target"]                                                  # doctest: +SKIP
    'Q6YK33'
    """
    import pandas as pd

    dtypes = hit_dtypes(tool)
    if not output.is_file() or output.stat().st_size == 0:
        return empty_hits(tool)
    # `astype` and not `read_csv(dtype=...)`: `dtype=` is annotated `dict[Hashable, Dtype]`,
    # and `dict` is invariant in its key, so no `dict[str, str]` satisfies it.
    return pd.read_csv(output, sep="\t", header=None, names=list(dtypes)).astype(dtypes)


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
) -> pd.DataFrame:
    """Search one amino-acid sequence against ``database`` and return the hits.

    One ``mmseqs easy-search``. The query FASTA and the tool's output both live and die
    inside a :meth:`~protein.external.MmseqsLikeTool.scratch_dir`, so a search leaves
    nothing behind and the frame is the answer.

    Parameters
    ----------
    sequence : str
        The residues to search with.
    database : SearchTarget or str
        What to search against: a **Database**, or the name of a registered one.
    query_name : str, default "query"
        The FASTA header this query is written under, which is what the ``query`` column
        reports.
    tool : protein.external.MmseqsLikeTool, optional
        The tool to drive. Defaults to :class:`~protein.external.Mmseqs`.
    sensitivity, evalue, max_seqs, threads : optional
        As :func:`search_flags`.
    extra : sequence of str, optional
        Further arguments, appended unread.

    Returns
    -------
    pandas.DataFrame
        The hits, :func:`hit_dtypes`' columns in MMseqs2's order — identity is ``pident``, a
        percentage. Empty with the same columns when nothing was found.

    Raises
    ------
    LookupError
        If ``database`` names nothing registered.
    protein.external.ToolNotFoundError
        If ``mmseqs`` is not installed.
    RuntimeError
        If the search exits non-zero. The message carries the tool's own output.

    Examples
    --------
    >>> hits = search("MKTAYIAKQRQISFVKSHFSRQ", "swissprot")   # doctest: +SKIP
    >>> hits.columns[:3].tolist()                              # doctest: +SKIP
    ['query', 'target', 'pident']
    """
    from protein.io import fasta

    tool = tool if tool is not None else Mmseqs()
    target = database_path(database)
    flags = search_flags(
        sensitivity=sensitivity,
        evalue=evalue,
        max_seqs=max_seqs,
        threads=threads,
        extra=extra,
    )
    with tool.scratch_dir("search") as work:
        query = work / "query.fasta"
        fasta.write_records(query, [(query_name, sequence)])
        output = work / "hits.tsv"
        tool.easy_search(query, target, output, extra=flags)
        return read_hits(output, tool)
